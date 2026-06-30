"""Fixture for the ADR-009 AST guard — NOT real code, do not import.

This file deliberately imports a concrete causal-LM class so the guard
test can prove the detector flags the violation. The leading underscore
in the filename keeps pytest from auto-collecting it as a test module.
The import is annotated ``noqa`` because ruff would otherwise flag it
as unused (F401) — that's the point: nothing should ever use it.
"""
from transformers import LlamaForCausalLM  # noqa: F401
