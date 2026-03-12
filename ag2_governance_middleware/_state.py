"""Budget state shared across a single execution run."""

import asyncio
from dataclasses import dataclass, field

# Sentinel key -- uses object identity (not equality) to avoid collision
# with any user-defined key in context.dependencies.
BUDGET_STATE_KEY: object = object()


@dataclass
class BudgetState:
    """
    Execution-scoped budget counters.

    Create a fresh instance per agent run and inject via:
        ctx.dependencies[BUDGET_STATE_KEY] = BudgetState(...)

    All mutation is protected by _lock (asyncio.Lock).
    """

    max_tokens: float
    max_tool_calls: int
    max_llm_calls: int
    consumed_tokens: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    blocked_llm_calls: int = 0
    blocked_tool_calls: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def token_exhausted(self) -> bool:
        """True when consumed_tokens >= max_tokens."""
        return self.consumed_tokens >= self.max_tokens

    @property
    def llm_exhausted(self) -> bool:
        """True when llm_calls >= max_llm_calls."""
        return self.llm_calls >= self.max_llm_calls

    @property
    def tool_exhausted(self) -> bool:
        """True when tool_calls >= max_tool_calls."""
        return self.tool_calls >= self.max_tool_calls

    async def try_consume_llm_call(self) -> bool:
        """Atomically check and consume one LLM call slot.

        Returns True if consumed, False if exhausted.
        """
        async with self._lock:
            if self.token_exhausted or self.llm_exhausted:
                self.blocked_llm_calls += 1
                return False
            self.llm_calls += 1
            return True

    async def try_consume_tool_call(self) -> bool:
        """Atomically check and consume one tool call slot.

        Returns True if consumed, False if exhausted.
        """
        async with self._lock:
            if self.tool_exhausted:
                self.blocked_tool_calls += 1
                return False
            self.tool_calls += 1
            return True
