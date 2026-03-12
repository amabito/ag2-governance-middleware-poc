"""Secret redaction middleware -- scrubs sensitive patterns before LLM sees them."""

import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from .base import BaseMiddleware, CallNext

logger = logging.getLogger(__name__)


class SecretRedactionMiddleware(BaseMiddleware):
    """
    Redacts secret patterns from text events before they reach the LLM.

    This middleware should be placed outermost (first in the middleware list)
    so that raw secrets never appear in downstream logs or processing.

    PoC limitation -- shallow only: only the top-level content field of
    ModelRequest events is scanned. Nested structures, binary payloads, tool
    arguments, and non-ModelRequest event types are not processed. Production
    use requires recursive traversal of all event fields.

    New event objects are created on redaction rather than mutating originals.

    Parameters
    ----------
    patterns:
        List of compiled regex patterns. Each match is replaced with
        the replacement string.
    replacement:
        Replacement text. Default: "[REDACTED]".
    """

    def __init__(
        self,
        patterns: list[re.Pattern[str]],
        replacement: str = "[REDACTED]",
    ) -> None:
        self._patterns = list(patterns)
        self._replacement = replacement

    def _redact_text(self, text: str) -> tuple[str, int]:
        """Apply all patterns to text. Return (redacted_text, replacement_count)."""
        count = 0
        for pattern in self._patterns:
            text, n = pattern.subn(self._replacement, text)
            count += n
        return text, count

    async def on_llm_call(
        self,
        call_next: CallNext,
        *messages: Any,
        ctx: Any,
        tools: Iterable[Any],
    ) -> None:
        """
        Redact secrets from ModelRequest events before forwarding.

        Shallow only: only ModelRequest.content is scanned.
        """
        # Import here to keep module importable without autogen installed.
        from autogen.beta.events import ModelRequest

        redacted_messages: list[Any] = []
        total_redacted = 0

        for event in messages:
            if isinstance(event, ModelRequest) and isinstance(event.content, str):
                new_content, count = self._redact_text(event.content)
                if count:
                    # PoC simplification: ModelRequest field preservation.
                    # ModelRequest uses EventMeta (not dataclass/Pydantic), so
                    # dataclasses.replace() and model_copy() are unavailable.
                    # Production should use the framework's copy mechanism.
                    event = ModelRequest(content=new_content)
                    total_redacted += count
            redacted_messages.append(event)

        if total_redacted:
            logger.info(
                "[Redaction] Redacted %d secret(s) in events before LLM call",
                total_redacted,
            )

        await call_next(*redacted_messages, ctx=ctx, tools=tools)

    async def on_tool_call(
        self,
        call_next: Callable[[Any, Any], Awaitable[None]],
        event: Any,
        ctx: Any,
    ) -> None:
        """Pass-through -- redaction applies to LLM calls only."""
        await call_next(event, ctx)
