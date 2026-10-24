"""tomato-agent: confidence-gated agentic diagnosis of tomato leaf diseases (v1.0, fully offline).

Agents (paper Section 3):
  CaptureQualityAgent  -> tomato_agent.quality
  DiagnosisAgent       -> tomato_agent.models (+ tomato_agent.calibration)
  VerificationAgent    -> tomato_agent.gate
  ExpertReferralAgent  -> tomato_agent.expert
The live loop that ties them together is tomato_agent.pipeline.
"""
from .classes import CLASSES, C2I, NUM_CLASSES

__version__ = "1.0.0"
__all__ = ["CLASSES", "C2I", "NUM_CLASSES", "__version__"]
