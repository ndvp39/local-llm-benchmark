"""on_prem_llm_lab — On-Premises LLM benchmark lab.

Public surface re-exported per PLAN §8 / constitution §13.2. CLI, notebooks,
and any future GUI MUST consume the SDK from here and MUST NOT reach into
:mod:`services`, :mod:`backends`, or :mod:`shared` directly (constitution §3.1).
"""

# CRITICAL: load `.env` BEFORE any child import — `transformers` (via
# `shared.automodel_factory`) reads `HF_HOME` / `TRANSFORMERS_CACHE` at
# import time and caches the resolved path. If we defer this to
# `cli/main.py`, HF downloads silently go to `C:\Users\<u>\.cache\...`
# even though `.env` says `D:/...`. `override=True` because the parent
# shell may pass `HF_HOME=""` which python-dotenv otherwise treats as
# "already set" and skips. See prompts_book §11.9 (ENOSPC incident).
from dotenv import load_dotenv as _load_dotenv

_load_dotenv(override=True)

from on_prem_llm_lab.sdk import (  # noqa: E402
    EnvironmentNotInitializedError,
    OnPremLlmSDK,
    PlumbingNotRunError,
    PlumbingStageError,
)
from on_prem_llm_lab.shared.version import __version__  # noqa: E402

__all__ = [
    "EnvironmentNotInitializedError",
    "OnPremLlmSDK",
    "PlumbingNotRunError",
    "PlumbingStageError",
    "__version__",
]
