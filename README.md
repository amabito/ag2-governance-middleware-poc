# AG2 Governance Middleware PoC

Shows that AG2.1 beta's middleware hooks are expressive enough for
runtime governance -- budget enforcement, tool denial, and secret redaction.

## Relationship to AG2.1 Beta

This PoC is built against the middleware design described in
[PR #2439](https://github.com/ag2ai/ag2/pull/2439) (AG2.1 beta).
Since #2439 is still in DRAFT and the middleware base class has not
been merged to the beta branch, this package includes a standalone
`base.py` that follows the same onion-chain / `call_next` pattern
(`on_llm_call`, `on_tool_call`).

Once the official `BaseMiddleware` is merged, these middleware classes
can be ported to inherit from it with minimal changes.

## Middleware

| Middleware | Hook | Behaviour |
|---|---|---|
| `SecretRedactionMiddleware` | `on_llm_call` | Regex-based redaction of secrets before LLM sees content |
| `PolicyDenyMiddleware` | `on_tool_call` | Deny-list and predicate-based tool call blocking |
| `SharedBudgetMiddleware` | `on_llm_call`, `on_tool_call` | Shared token/call budget with short-circuit on exhaustion |

Recommended ordering (outermost first):
**Redaction -> Policy -> Budget**

## Setup

Clone this repo next to the [ag2](https://github.com/ag2ai/ag2) repo
(`ag2.1-beta` branch) so that `autogen.beta` is importable:

```
parent/
  ag2/                              # ag2.1-beta branch
  ag2-governance-middleware-poc/     # this repo
```

## Running the demo

```bash
cd ag2-governance-middleware-poc
python demo.py
```

No API keys required. All tests use mock LLM and mock Context.

Adversarial tests (15 tests covering TOCTOU, corrupted input, boundary
conditions, predicate failures, full-stack integration):

```bash
python test_adversarial.py
```

## PoC Limitations

- **Redaction is shallow**: only top-level `ModelRequest.content` is scanned.
  Nested structures and non-text payloads are not processed.
- **Token tracking is approximate**: `consumed_tokens` is not auto-incremented
  after LLM calls. The AG2.1 beta `LLMClient` emits `ModelResponse` via
  `ctx.send()` (fire-and-forget), so the response is not accessible in
  the middleware. Callers can update `consumed_tokens` directly.
- **No circuit breaker**: planned for a second iteration.
