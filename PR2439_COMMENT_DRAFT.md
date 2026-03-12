I built a governance middleware PoC on top of the `on_llm_call` / `on_tool_call` + `call_next` onion chain from this PR.

Three middleware in the stack:

- **SharedBudgetMiddleware** -- enforces LLM call count, tool call count, and token limits across an execution run. Short-circuits with a synthetic `ModelResponse` or `ToolError` when budget is exhausted.
- **PolicyDenyMiddleware** -- deny-list + predicate-based tool call blocking. Fail-closed: predicate exceptions result in DENY.
- **SecretRedactionMiddleware** -- regex-based redaction on `ModelRequest.content` before the LLM sees it. Non-destructive (creates new event objects).

Repo: https://github.com/amabito/ag2-governance-middleware-poc

`python demo.py` runs 4 checks with mock context, no API keys needed. There's also `test_adversarial.py` with 15 tests covering TOCTOU races, corrupted inputs, and boundary conditions.

One open question: sharing execution-scoped state across middleware (e.g. a budget counter that both `on_llm_call` and `on_tool_call` read/write) currently goes through `ctx.dependencies` with a sentinel key. A standard pattern for this in the middleware API would make it easier to compose middleware from different authors.
