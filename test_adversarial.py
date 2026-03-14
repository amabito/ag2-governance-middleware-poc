#!/usr/bin/env python3
"""
Adversarial tests for AG2 Governance Middleware PoC.

Tests attack vectors: TOCTOU races, None/corrupted inputs, boundary
conditions, predicate failures, and full-stack integration.
"""

import asyncio
import re
import sys
import os
from dataclasses import dataclass, field
from typing import Any

# -- Path setup ---------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AG2_ROOT = os.path.dirname(_THIS_DIR)
if _AG2_ROOT not in sys.path:
    sys.path.insert(0, _AG2_ROOT)

from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolCalls,
    ToolError,
)

from ag2_governance_middleware import (
    BUDGET_STATE_KEY,
    BudgetState,
    PolicyDenyMiddleware,
    SecretRedactionMiddleware,
    SharedBudgetMiddleware,
    build_middleware_client,
    build_middleware_tool_chain,
)


# -- Mock Context -------------------------------------------------------------

@dataclass
class MockContext:
    sent_events: list[BaseEvent] = field(default_factory=list)
    dependencies: dict[Any, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    prompt: list[str] = field(default_factory=list)

    async def send(self, event: BaseEvent) -> None:
        self.sent_events.append(event)


# -- Mock ToolCall with controllable attributes --------------------------------

@dataclass
class FakeToolCall:
    """Allows setting name/arguments to arbitrary values including None."""
    id: str = "tc-adv"
    name: Any = "test_tool"
    arguments: Any = "{}"


# -- Test runner ---------------------------------------------------------------

async def run_test(name: str, coro) -> bool:
    print(f"\n--- {name} ---")
    try:
        result = await coro
        label = "PASS" if result else "FAIL"
        print(f"  [{label}] {name}")
        return bool(result)
    except Exception as exc:
        print(f"  [FAIL] {name} -- exception: {exc!r}")
        return False


# =============================================================================
# Category 1: Corrupted Input
# =============================================================================

async def test_policy_event_name_none() -> bool:
    """event.name=None must be denied (fail-closed), not bypass deny list."""
    mw = PolicyDenyMiddleware(denied_tools={"shell_exec"})
    ctx = MockContext()
    event = FakeToolCall(name=None, arguments='{}')

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    error_emitted = any(isinstance(e, ToolError) for e in ctx.sent_events)
    print(f"    call_next={call_next_invoked}, error={error_emitted}")
    return (not call_next_invoked) and error_emitted


async def test_policy_event_arguments_none() -> bool:
    """event.arguments=None must not crash json.loads (TypeError caught)."""
    mw = PolicyDenyMiddleware(
        denied_predicates=[lambda name, args: args.get("danger", False)]
    )
    ctx = MockContext()
    event = FakeToolCall(name="safe_tool", arguments=None)

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    # Should proceed (predicate sees empty args, returns False)
    print(f"    call_next={call_next_invoked}")
    return call_next_invoked


async def test_policy_event_arguments_binary() -> bool:
    """Binary (non-JSON) arguments must not crash -- uses empty args."""
    mw = PolicyDenyMiddleware(
        denied_predicates=[lambda name, args: False]
    )
    ctx = MockContext()
    event = FakeToolCall(name="tool", arguments=b"\x00\xff\xfe")

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)
    print(f"    call_next={call_next_invoked}")
    return call_next_invoked


async def test_redaction_content_none() -> bool:
    """ModelRequest with content=None must not crash redaction middleware."""
    mw = SecretRedactionMiddleware(
        patterns=[re.compile(r"SECRET")]
    )

    received: list[Any] = []

    async def capture(*messages: Any, ctx: Any, tools: Any) -> None:
        received.extend(messages)

    client = build_middleware_client(capture, [mw])

    # ModelRequest uses EventMeta, not frozen dataclass, so attribute
    # assignment works. If it ever becomes frozen, use FakeToolCall-style mock.
    event = ModelRequest(content="no secret here")
    event.content = None  # Force None after construction

    ctx = MockContext()
    await client(event, ctx=ctx, tools=[])

    print(f"    received={len(received)} events (no crash)")
    return len(received) == 1


async def test_redaction_content_non_string() -> bool:
    """ModelRequest with non-str content (int) must pass through unchanged."""
    mw = SecretRedactionMiddleware(
        patterns=[re.compile(r"\d+")]
    )

    received: list[Any] = []

    async def capture(*messages: Any, ctx: Any, tools: Any) -> None:
        received.extend(messages)

    client = build_middleware_client(capture, [mw])

    event = ModelRequest(content="hello")
    event.content = 12345  # Force int after construction

    ctx = MockContext()
    await client(event, ctx=ctx, tools=[])

    # Should pass through without crash, content unchanged
    ok = len(received) == 1 and received[0].content == 12345
    print(f"    received content={received[0].content if received else 'NONE'}")
    return ok


# =============================================================================
# Category 2: Predicate Failure (fail-closed)
# =============================================================================

async def test_policy_predicate_exception_denies() -> bool:
    """Predicate that raises must result in DENY (fail-closed)."""
    def exploding_predicate(name: str, args: dict) -> bool:
        raise RuntimeError("I broke")

    mw = PolicyDenyMiddleware(denied_predicates=[exploding_predicate])
    ctx = MockContext()
    event = ToolCall(id="tc-x", name="innocent_tool", arguments='{}')

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    error_emitted = any(isinstance(e, ToolError) for e in ctx.sent_events)
    print(f"    call_next={call_next_invoked}, error={error_emitted}")
    return (not call_next_invoked) and error_emitted


async def test_policy_first_predicate_ok_second_explodes() -> bool:
    """First predicate passes, second raises -- must still DENY."""
    def ok_predicate(name: str, args: dict) -> bool:
        return False  # Allow

    def bad_predicate(name: str, args: dict) -> bool:
        raise ValueError("boom")

    mw = PolicyDenyMiddleware(denied_predicates=[ok_predicate, bad_predicate])
    ctx = MockContext()
    event = ToolCall(id="tc-y", name="tool_y", arguments='{}')

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    error_emitted = any(isinstance(e, ToolError) for e in ctx.sent_events)
    print(f"    call_next={call_next_invoked}, error={error_emitted}")
    return (not call_next_invoked) and error_emitted


# =============================================================================
# Category 3: Boundary Conditions (N-1, N, N+1)
# =============================================================================

async def test_budget_llm_exactly_at_limit() -> bool:
    """LLM call at exact limit (calls=max) must be blocked."""
    state = BudgetState(max_tokens=10_000, max_tool_calls=10, max_llm_calls=2)

    # Consume 2 calls (the max)
    assert await state.try_consume_llm_call()  # call 1
    assert await state.try_consume_llm_call()  # call 2

    # 3rd must be blocked
    blocked = not await state.try_consume_llm_call()
    print(f"    llm_calls={state.llm_calls}, blocked={blocked}, blocked_count={state.blocked_llm_calls}")
    return blocked and state.blocked_llm_calls == 1


async def test_budget_tool_boundary_n_minus_1() -> bool:
    """Tool call at N-1 must succeed, at N must succeed, at N+1 must block."""
    state = BudgetState(max_tokens=10_000, max_tool_calls=3, max_llm_calls=10)

    results = []
    for i in range(5):
        results.append(await state.try_consume_tool_call())

    # First 3 succeed, next 2 blocked
    expected = [True, True, True, False, False]
    print(f"    results={results}, expected={expected}")
    return results == expected


async def test_budget_token_exhaustion_blocks_llm() -> bool:
    """consumed_tokens >= max_tokens must block LLM calls even if llm_calls < max."""
    state = BudgetState(max_tokens=100.0, max_tool_calls=10, max_llm_calls=10)
    state.consumed_tokens = 100.0  # Exactly at limit

    blocked = not await state.try_consume_llm_call()
    print(f"    tokens={state.consumed_tokens}/{state.max_tokens}, blocked={blocked}")
    return blocked


async def test_budget_token_just_under_limit() -> bool:
    """consumed_tokens just under max must still allow LLM call."""
    state = BudgetState(max_tokens=100.0, max_tool_calls=10, max_llm_calls=10)
    state.consumed_tokens = 99.9

    allowed = await state.try_consume_llm_call()
    print(f"    tokens={state.consumed_tokens}/{state.max_tokens}, allowed={allowed}")
    return allowed


# =============================================================================
# Category 4: Concurrent Access (TOCTOU)
# =============================================================================

async def test_concurrent_llm_budget_consumption() -> bool:
    """Multiple concurrent try_consume_llm_call must not exceed max."""
    state = BudgetState(max_tokens=10_000, max_tool_calls=10, max_llm_calls=5)

    async def consume() -> bool:
        return await state.try_consume_llm_call()

    # Launch 20 concurrent consumers for 5 slots
    results = await asyncio.gather(*[consume() for _ in range(20)])

    successes = sum(1 for r in results if r)
    print(f"    successes={successes}/5 expected, llm_calls={state.llm_calls}")
    return successes == 5 and state.llm_calls == 5


async def test_concurrent_tool_budget_consumption() -> bool:
    """Multiple concurrent try_consume_tool_call must not exceed max."""
    state = BudgetState(max_tokens=10_000, max_tool_calls=3, max_llm_calls=10)

    async def consume() -> bool:
        return await state.try_consume_tool_call()

    results = await asyncio.gather(*[consume() for _ in range(15)])

    successes = sum(1 for r in results if r)
    print(f"    successes={successes}/3 expected, tool_calls={state.tool_calls}")
    return successes == 3 and state.tool_calls == 3


# =============================================================================
# Category 5: Full-Stack Integration
# =============================================================================

async def test_full_stack_redaction_then_policy_then_budget() -> bool:
    """
    Full middleware stack: Redaction -> Policy -> Budget.
    Secret in tool arguments: redaction only applies to LLM calls, so
    policy and budget should still see the original tool call.
    """
    redaction_mw = SecretRedactionMiddleware(
        patterns=[re.compile(r"sk-[a-zA-Z0-9]+")]
    )
    policy_mw = PolicyDenyMiddleware(denied_tools={"dangerous_tool"})
    budget_state = BudgetState(max_tokens=10_000, max_tool_calls=10, max_llm_calls=10)

    stack = [redaction_mw, policy_mw, SharedBudgetMiddleware()]
    ctx = MockContext(dependencies={BUDGET_STATE_KEY: budget_state})

    # Test 1: LLM call with secret -- should be redacted
    received_content: list[str] = []

    async def capture_llm(*messages: Any, ctx: Any, tools: Any) -> None:
        for msg in messages:
            if isinstance(msg, ModelRequest):
                received_content.append(msg.content)

    llm_client = build_middleware_client(capture_llm, stack)
    await llm_client(
        ModelRequest(content="Key is sk-abc123xyz"),
        ctx=ctx,
        tools=[],
    )

    secret_redacted = "sk-abc123xyz" not in received_content[0] if received_content else False

    # Test 2: Tool call to denied tool -- should be blocked by policy
    tool_chain = build_middleware_tool_chain(
        lambda ev, ctx: asyncio.sleep(0),
        stack,
    )

    await tool_chain(
        ToolCall(id="tc-fs", name="dangerous_tool", arguments='{}'),
        ctx,
    )

    policy_blocked = any(
        isinstance(e, ToolError) and e.name == "dangerous_tool"
        for e in ctx.sent_events
    )

    # Test 3: Allowed tool call -- should consume budget
    await tool_chain(
        ToolCall(id="tc-ok", name="safe_tool", arguments='{}'),
        ctx,
    )

    budget_consumed = budget_state.tool_calls == 1  # Only safe_tool consumed

    print(
        f"    secret_redacted={secret_redacted}, "
        f"policy_blocked={policy_blocked}, "
        f"budget_consumed={budget_consumed}"
    )
    return secret_redacted and policy_blocked and budget_consumed


async def test_policy_deny_list_and_predicate_combined() -> bool:
    """Tool in deny list is blocked even if predicate would allow it."""
    mw = PolicyDenyMiddleware(
        denied_tools={"blocked_tool"},
        denied_predicates=[lambda name, args: False],  # Would allow
    )
    ctx = MockContext()
    event = ToolCall(id="tc-combo", name="blocked_tool", arguments='{}')

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    error_emitted = any(isinstance(e, ToolError) for e in ctx.sent_events)
    print(f"    call_next={call_next_invoked}, error={error_emitted}")
    return (not call_next_invoked) and error_emitted


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    tests = [
        # Category 1: Corrupted Input
        ("Corrupted: event.name=None", test_policy_event_name_none),
        ("Corrupted: event.arguments=None", test_policy_event_arguments_none),
        ("Corrupted: event.arguments=bytes", test_policy_event_arguments_binary),
        ("Corrupted: redaction content=None", test_redaction_content_none),
        ("Corrupted: redaction content=int", test_redaction_content_non_string),
        # Category 2: Predicate Failure
        ("Predicate: exception -> DENY", test_policy_predicate_exception_denies),
        ("Predicate: 2nd explodes -> DENY", test_policy_first_predicate_ok_second_explodes),
        # Category 3: Boundary Conditions
        ("Boundary: LLM at exact limit", test_budget_llm_exactly_at_limit),
        ("Boundary: tool N-1/N/N+1", test_budget_tool_boundary_n_minus_1),
        ("Boundary: token exhaustion", test_budget_token_exhaustion_blocks_llm),
        ("Boundary: token just under", test_budget_token_just_under_limit),
        # Category 4: Concurrent Access
        ("Concurrent: LLM budget race", test_concurrent_llm_budget_consumption),
        ("Concurrent: tool budget race", test_concurrent_tool_budget_consumption),
        # Category 5: Full-Stack Integration
        ("Integration: full stack", test_full_stack_redaction_then_policy_then_budget),
        ("Integration: deny list + predicate", test_policy_deny_list_and_predicate_combined),
    ]

    results: list[bool] = []
    for name, test_fn in tests:
        result = await run_test(name, test_fn())
        results.append(result)

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"Adversarial: {passed}/{total} passed")
    if passed == total:
        print("ALL ADVERSARIAL TESTS PASS")
        sys.exit(0)
    else:
        failed_names = [
            name for (name, _), r in zip(tests, results) if not r
        ]
        print(f"FAILED: {failed_names}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
