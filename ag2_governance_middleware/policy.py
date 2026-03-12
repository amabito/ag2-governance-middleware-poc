"""Policy deny middleware -- blocks tool calls matching a deny list or predicates."""

import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from .base import BaseMiddleware, CallNext
from ._helpers import send_tool_error

logger = logging.getLogger(__name__)


class PolicyViolationError(Exception):
    """Raised when a tool call is denied by governance policy."""


class PolicyDenyMiddleware(BaseMiddleware):
    """
    Denies tool calls that appear in a deny list or match predicate functions.

    LLM calls are passed through unchanged. Only on_tool_call() is active.

    Parameters
    ----------
    denied_tools:
        Set of tool names that are unconditionally denied.
    denied_predicates:
        List of callables with signature (tool_name: str, args: dict) -> bool.
        If any predicate returns True, the tool call is denied. Predicate
        exceptions are treated as DENY (fail-closed) with a warning logged.
    """

    def __init__(
        self,
        denied_tools: set[str] | None = None,
        denied_predicates: list[Callable[[str, dict[str, Any]], bool]] | None = None,
    ) -> None:
        self._denied_tools: set[str] = denied_tools or set()
        self._denied_predicates: list[Callable[[str, dict[str, Any]], bool]] = denied_predicates or []

    async def on_llm_call(
        self,
        call_next: CallNext,
        *messages: Any,
        ctx: Any,
        tools: Iterable[Any],
    ) -> None:
        """Pass-through -- policy only applies to tool calls."""
        await call_next(*messages, ctx=ctx, tools=tools)

    async def on_tool_call(
        self,
        call_next: Callable[[Any, Any], Awaitable[None]],
        event: Any,
        ctx: Any,
    ) -> None:
        """
        Evaluate deny list and predicates. Emit ToolError on denial without
        invoking call_next.
        """
        tool_name = event.name
        if not isinstance(tool_name, str):
            logger.warning(
                "[Policy] Tool event has non-str name %r -- treating as DENY",
                tool_name,
            )
            err = PolicyViolationError(
                "Tool event has invalid name (non-str) -- denied by governance policy"
            )
            await send_tool_error(ctx, event, err)
            return

        denied = tool_name in self._denied_tools

        if not denied and self._denied_predicates:
            try:
                parsed = json.loads(event.arguments)
                args: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "[Policy] Failed to parse arguments for tool '%s' -- using empty args",
                    tool_name,
                )
                args = {}

            for predicate in self._denied_predicates:
                try:
                    if predicate(tool_name, args):
                        denied = True
                        break
                except Exception as exc:
                    logger.warning(
                        "[Policy] Predicate %r raised for tool '%s': %r -- treating as DENY",
                        predicate,
                        tool_name,
                        exc,
                    )
                    denied = True
                    break

        if denied:
            logger.warning("[Policy] Tool '%s' DENIED by policy", tool_name)
            err = PolicyViolationError(
                f"Tool '{tool_name}' is denied by governance policy"
            )
            await send_tool_error(ctx, event, err)
            return

        await call_next(event, ctx)
