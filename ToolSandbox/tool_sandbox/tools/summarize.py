# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""A generic tool for summarizing the current task state."""

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.utils import register_as_tool
from tool_sandbox.common.validators import typechecked


@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def summarize_the_task(summary: str) -> str:
    """Write a summary of the current state to aid decision-making.

    Args:
        summary: Summary of the current state, including environment and user info.

    Returns:
        A standardized string containing the summary content.
    """
    message = f"{str(summary).strip()}"
    print(f"[SummaryTool] {message}")
    return message


