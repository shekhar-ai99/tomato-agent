#!/usr/bin/env python3
"""
Export one trained model for on-device use and measure edge feasibility (paper Section 5.5).

  * ONNX FP32 export + numerical parity check against PyTorch
  * ONNX INT8 (static, QDQ) quantisation calibrated on validation photos + top-1 agreement check
  * CPU latency per image (PyTorch, ONNX FP32, ONNX INT8) and file sizes
  * patrasathi_config.json: classes, preprocessing, temperature, gate thresholds and quality
    thresholds in one file, so the phone makes exactly the decisions the paper evaluates

Outputs in --out (default export/<backbone>_seed<k>/):
  model_fp32.onnx, model_int8.onnx, patrasathi_config.json, export_report.json, advice.json, retake_tips.json
--android-assets <dir> also copies the phone files there (e.g. tomato-agent-andriod/app/src/main/assets/model).

Usage:
  python scripts/export_mobile.py --run runs/efficientnet_b0/seed0 --pv-root PlantVillage-Dataset --pd-root PlantDoc-Dataset
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from tomato_agent import __version__
from tomato_agent.classes import CLASSES
from tomato_agent.pipeline import gate_for
from tomato_agent.io_utils import add_data_args, eval_set, read_json, write_json
from tomato_agent.models import LoadedRun
from tomato_agent.quality import FEATURES, WORK_SIZE

SHARED = Path(__file__).resolve().parents[1] / "shared"


MIN_INT8_AGREEMENT = 0.95  # INT8 must give the same top-1 as fp32 on at least 95% of the check photos


def preprocess(run, pil):
    return run.tf(pil.convert("RGB")).unsqueeze(0).numpy()


def export_onnx(model, size, path):
    model = model.cpu().eval()
    x = torch.randn(1, 3, size, size)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, (x,), str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, (x,), str(path), **kw)


def latency(fn, x, n=30, warm=5):
    for _ in range(warm):
        fn(x)
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs/<backbone>/seed<k>")
    add_data_args(ap)
    ap.add_argument("--quality", default="artifacts/quality_thresholds.json")
    ap.add_argument("--thresholds", default="results/thresholds.json")
    ap.add_argument("--cost", default="10:1")
    ap.add_argument("--temperature", default="pv", choices=["pv", "pd", "none"])
    ap.add_argument("--app-retake-budget", type=int, default=2,
                    help="retakes allowed on the phone (the paper evaluates 1; the plan allows up to 3 tries)")
    ap.add_argument("--calib-images", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--android-assets", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from PIL import Image

    run = LoadedRun(args.run, torch.device("cpu"))
    out = Path(args.out or f"export/{run.backbone}_seed{run.seed}")
    out.mkdir(parents=True, exist_ok=True)
    fp32, pre, int8 = out / "model_fp32.onnx", out / "model_pre.onnx", out / "model_int8.onnx"
    int8.unlink(missing_ok=True)  # never leave an INT8 file from an earlier export next to a new fp32 model
    export_onnx(run.model, run.img_size, fp32)
    onnx.checker.check_model(str(fp32))

    # calibration / check photos: validation sets only, never test
    imgs = []
    for s in ("pv_val", "pd_calib"):
        root, rows = eval_set(args, s)
        rng = np.random.default_rng(0)
        for i in rng.permutation(len(rows))[: args.calib_images // 2]:
            with Image.open(root / rows[i]["path"]) as im:
                imgs.append(im.convert("RGB").copy())
    xs = [preprocess(run, im) for im in imgs]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": x} for x in xs])

        def get_next(self):
            return next(self.it, None)

    try:
        quant_pre_process(str(fp32), str(pre))
        src = pre
    except Exception as e:  # pre-processing is an optimisation; quantise the raw graph if it fails
        print(f"quant_pre_process skipped: {e}")
        src = fp32

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s32 = ort.InferenceSession(str(fp32), so, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = np.concatenate([run.model(torch.from_numpy(x)).numpy() for x in xs])
    o32 = np.concatenate([s32.run(None, {"input": x})[0] for x in xs])

    # Static INT8 can break some networks (EfficientNet's SiLU/squeeze-excite layers in particular), so try a few
    # calibration settings, keep the one that agrees best with the fp32 model, and ship INT8 only if it is good enough.
    candidates = [("minmax", CalibrationMethod.MinMax, {}),
                  ("percentile", CalibrationMethod.Percentile, {"CalibPercentile": 99.999}),
                  ("entropy", CalibrationMethod.Entropy, {})]
    tried, best, s8 = {}, None, None
    for name, method, extra in candidates:
        cand = out / f"model_int8_{name}.onnx"
        try:
            quantize_static(str(src), str(cand), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            calibrate_method=method, extra_options=extra)
            sess = ort.InferenceSession(str(cand), so, providers=["CPUExecutionProvider"])
            o = np.concatenate([sess.run(None, {"input": x})[0] for x in xs])
        except Exception as e:
            print(f"int8 calibration '{name}' failed: {e}")
            cand.unlink(missing_ok=True)
            continue
        agree = float((o.argmax(1) == ref.argmax(1)).mean())
        tried[name] = agree
        print(f"int8 calibration '{name}': top-1 agreement with fp32 = {agree:.3f}")
        if best is None or agree > tried[best[0]]:
            if best is not None:
                best[1].unlink(missing_ok=True)
            best, s8 = (name, cand), sess
        else:
            cand.unlink(missing_ok=True)
    pre.unlink(missing_ok=True)
    int8_ok = best is not None and tried[best[0]] >= MIN_INT8_AGREEMENT
    if best is not None:
        if int8_ok:
            best[1].replace(int8)
        else:
            best[1].unlink(missing_ok=True)
            s8 = None
            print(f"WARNING: best INT8 agreement {tried[best[0]]:.3f} < {MIN_INT8_AGREEMENT}; not shipping INT8, "
                  "the phone config uses the fp32 model.")
    else:
        print("WARNING: INT8 quantisation failed for every setting; the phone config uses the fp32 model.")

    x0 = xs[0]
    report = {
        "backbone": run.backbone, "seed": run.seed, "img_size": run.img_size, "n_check_images": len(xs),
        "fp32_max_abs_diff_vs_torch": float(np.abs(o32 - ref).max()),
        "fp32_top1_agreement": float((o32.argmax(1) == ref.argmax(1)).mean()),
        "int8_top1_agreement_by_calibration": tried,
        "int8_top1_agreement": tried[best[0]] if best else None,
        "int8_calibration_used": best[0] if int8_ok else None,
        "int8_shipped": int8_ok,
        "size_mb": {"fp32": fp32.stat().st_size / 1e6, "int8": int8.stat().st_size / 1e6 if int8_ok else None},
        "latency_ms_cpu_batch1": {
            "torch": latency(lambda x: run.model(torch.from_numpy(x)), x0),
            "onnx_fp32": latency(lambda x: s32.run(None, {"input": x}), x0),
            "onnx_int8": latency(lambda x: s8.run(None, {"input": x}), x0) if int8_ok else None},
        "threads": so.intra_op_num_threads,
    }
    write_json(out / "export_report.json", report)

    q = read_json(args.quality)["thresholds"] if Path(args.quality).exists() else None
    th = read_json(args.thresholds).get("recommended", {}) if Path(args.thresholds).exists() else {}
    gate = th.get(args.cost)
    if q is None or gate is None:
        print("WARNING: quality or gate thresholds missing; the phone config gets placeholders. "
              "Run set_quality_thresholds.py and run_policy.py first for the real values.")
    T = run.temperature(args.temperature) if args.temperature != "none" else 1.0
    th_all = read_json(args.thresholds) if Path(args.thresholds).exists() else None
    calibrations = {}
    for t in ("pv", "pd"):
        g = gate_for(th_all, args.cost, t, run.backbone) if th_all and th_all.get("recommended") else None
        calibrations[t] = {"temperature": run.temperature(t),
                           "gate": None if g is None else {"tau_accept": g.tau_accept, "tau_defer": g.tau_defer,
                                                           "paper_retake_budget": g.retake_budget}}
    config = {
        "app": "PatraSathi", "config_version": 1, "tomato_agent_version": __version__,
        "model": {"file": "model_int8.onnx" if int8_ok else "model_fp32.onnx",
                  "fallback_file": "model_fp32.onnx" if int8_ok else None, "backbone": run.backbone,
                  "seed": run.seed, "input_name": "input", "output_name": "logits",
                  "input_size": run.img_size, "resize": "square_bilinear_no_crop",
                  "mean": list(run.data_cfg["mean"]), "std": list(run.data_cfg["std"])},
        "classes": CLASSES,
        "temperature": T, "temperature_fitted_on": args.temperature,
        "gate": {"tau_accept": (gate or {}).get("tau_accept", 0.90), "tau_defer": (gate or {}).get("tau_defer", 0.50),
                 "retake_budget": args.app_retake_budget, "cost_ratio": args.cost,
                 "paper_retake_budget": (gate or {}).get("retake_budget", 1)},
        # both calibrations from the paper: "pv" (temperature fitted on PlantVillage) and "pd" (fitted on the PlantDoc
        # hold-out); "temperature" and "gate" above are the one chosen with --temperature
        "calibrations": calibrations,
        "quality": {"work_size": WORK_SIZE, "features": FEATURES, "thresholds": q or {}},
        "placeholders": q is None or gate is None,
    }
    write_json(out / "patrasathi_config.json", config)
    for f in ("advice.json", "retake_tips.json"):
        shutil.copy(SHARED / f, out / f)
    print(f"Exported to {out}/")
    for k, v in report.items():
        print(f"  {k}: {v}")
    if args.android_assets:
        dst = Path(args.android_assets)
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("model_int8.onnx", "model_fp32.onnx", "patrasathi_config.json", "advice.json", "retake_tips.json"):
            if (out / f).exists():
                shutil.copy(out / f, dst / f)
        print(f"Copied phone files to {dst}/")


if __name__ == "__main__":
    main()
