"""SDK subpackage — re-exports the public facade, result types, and ADR-016 error."""

from on_prem_llm_lab.sdk.sdk import (
    EnvironmentNotInitializedError,
    InitEnvResult,
    OnPremLlmSDK,
)

__all__ = ["EnvironmentNotInitializedError", "InitEnvResult", "OnPremLlmSDK"]
