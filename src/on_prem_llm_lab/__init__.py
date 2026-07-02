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

# Monkey-patch AirLLM utils to use CPU device when torch.cuda.is_available()
# is False. Upstream airllm v2.11 hardcodes device="cuda" and .cuda() calls
# in utils.py:92,94,105-107,162,169 for its NF4/int8 compression paths;
# bitsandbytes 0.49+ has a working CPU backend, so the only real blocker is
# AirLLM's hardcoded device pin. See shared/airllm_cpu_patch.py.
from on_prem_llm_lab.shared.airllm_cpu_patch import apply_cpu_patch as _apply  # noqa: E402

_apply()

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
