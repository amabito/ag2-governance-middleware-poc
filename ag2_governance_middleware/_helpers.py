"""Shared helpers for governance middleware."""

from typing import Any


async def send_tool_error(ctx: Any, event: Any, error: Exception) -> None:
    """Emit a ToolError to the context stream.

    Centralizes ToolError construction to avoid duplication across
    budget and policy middleware.
    """
    from autogen.beta.events import ToolError

    await ctx.send(
        ToolError(
            parent_id=getattr(event, "id", None),
            name=str(getattr(event, "name", "<unknown>")),
            content=repr(error),
            error=error,
        )
    )
