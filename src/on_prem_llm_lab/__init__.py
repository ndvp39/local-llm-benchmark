"""on_prem_llm_lab — On-Premises LLM benchmark lab.

Public surface re-exported per PLAN §8 / constitution §13.2. CLI, notebooks,
and any future GUI MUST consume the SDK from here and MUST NOT reach into
:mod:`services`, :mod:`backends`, or :mod:`shared` directly (constitution §3.1).
"""

from on_prem_llm_lab.sdk import EnvironmentNotInitializedError, OnPremLlmSDK
from on_prem_llm_lab.shared.version import __version__

__all__ = ["EnvironmentNotInitializedError", "OnPremLlmSDK", "__version__"]
