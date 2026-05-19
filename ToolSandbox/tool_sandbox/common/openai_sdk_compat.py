# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Helpers for constructing OpenAI Python SDK clients across roles."""

from __future__ import annotations

import os


def openai_api_key_or_placeholder() -> str:
    """The OpenAI SDK requires ``api_key``; vLLM/Ollama typically accept a dummy value."""
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_COMPAT_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or "EMPTY"
    )


def openai_compatible_base_url() -> str:
    """OpenAI-compatible server root (vLLM default includes ``/v1``)."""
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("VLLM_OPENAI_BASE_URL")
        or os.environ.get("QWEN_BASE_URL")
        or "http://localhost:8000/v1"
    )
