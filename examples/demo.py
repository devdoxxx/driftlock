"""
What:     End-to-end feature tour — tracking, optimization, budget guardrails,
          cache, and context tags — using a patched fake OpenAI backend.
Requires: Nothing. No API key, no network calls, no cost.
Run:      python examples/demo.py
"""

import json
from unittest.mock import MagicMock, patch

import driftlock
from driftlock import (
    BudgetExceededError,
    CacheConfig,
    DriftlockClient,
    DriftlockConfig,
    OptimizationConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(content="OK.", prompt_tokens=80, completion_tokens=30, model="gpt-4o-mini"):
    """Build a minimal ChatCompletion-shaped mock."""
    r = MagicMock()
    r.model = model
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    r.usage.total_tokens = prompt_tokens + completion_tokens
    r.choices[0].message.content = content
    return r


def _section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _show_recent(client: DriftlockClient, n: int = 1) -> None:
    for rec in client.recent_calls(limit=n):
        keys = ["model", "prompt_tokens", "completion_tokens", "latency_ms",
                "estimated_cost_usd", "cache_hit", "labels", "warnings"]
        if rec.get("cache_hit"):
            keys += ["tokens_saved_prompt", "tokens_saved_completion", "estimated_savings_usd"]
        print(json.dumps({k: rec[k] for k in keys if k in rec}, indent=2))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    # Single shared config: human-readable logs, in-memory SQLite (no files created)
    config = DriftlockConfig(log_json=False, db_path=":memory:")

    with patch("driftlock.client.OpenAI") as MockOpenAI:
        mock_openai = MockOpenAI.return_value

        # ------------------------------------------------------------------ #
        # 1. Basic tracking — no optimization, no cache
        # ------------------------------------------------------------------ #
        _section("1 / 5  Basic call — token tracking + cost estimation")

        mock_openai.chat.completions.create.return_value = _fake_response(
            "The capital of France is Paris.",
            prompt_tokens=42,
            completion_tokens=8,
        )

        client = DriftlockClient(api_key="demo", config=config)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            _dl_endpoint="geography",
        )
        print(f"\nResponse: {response.choices[0].message.content}")
        print("\nStored record:")
        _show_recent(client)

        # ------------------------------------------------------------------ #
        # 2. Optimization: prompt trimming + output cap
        # ------------------------------------------------------------------ #
        _section("2 / 5  Optimization — long history trimmed to fit 200-token budget")

        # Build a long conversation history (10 turns)
        long_history = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(1, 11):
            long_history += [
                {"role": "user",      "content": f"Question number {i}: tell me something interesting."},
                {"role": "assistant", "content": f"Interesting fact number {i}: the universe is vast."},
            ]
        long_history.append({"role": "user", "content": "Summarise everything so far."})

        mock_openai.chat.completions.create.return_value = _fake_response(
            "Here is a summary...", prompt_tokens=180, completion_tokens=40
        )

        opt_client = DriftlockClient(
            api_key="demo",
            config=config,
            optimization=OptimizationConfig(
                max_prompt_tokens=200,
                keep_last_n_messages=4,
                default_max_output_tokens=256,
            ),
        )
        opt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=long_history,
            _dl_endpoint="summarise",
        )
        print(f"\nOriginal message count : {len(long_history)}")
        print("(Optimization details visible in the log line above: saved=Xtok | opts=[...])")

        # ------------------------------------------------------------------ #
        # 3. Budget guardrail — raise, then fallback
        # ------------------------------------------------------------------ #
        _section("3 / 5  Budget guardrail — raise then fallback to cheaper model")

        guard_client = DriftlockClient(
            api_key="demo",
            config=config,
            optimization=OptimizationConfig(
                max_cost_per_request_usd=0.000001,   # $0.000001 — always exceeded
                budget_exceeded_action="raise",
            ),
        )

        print("\n[raise mode]")
        try:
            guard_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "expensive call"}],
            )
        except BudgetExceededError as e:
            print(f"Caught BudgetExceededError: {e}")

        print("\n[fallback mode — switches to gpt-4o-mini automatically]")
        mock_openai.chat.completions.create.return_value = _fake_response(
            "Cheaper answer.", model="gpt-4o-mini"
        )

        fallback_client = DriftlockClient(
            api_key="demo",
            config=config,
            optimization=OptimizationConfig(
                max_cost_per_request_usd=0.000001,
                budget_exceeded_action="fallback",
                fallback_model="gpt-4o-mini",
            ),
        )
        fallback_client.chat.completions.create(
            model="gpt-4o",   # ← requested gpt-4o
            messages=[{"role": "user", "content": "expensive call"}],
        )
        rec = fallback_client.recent_calls(limit=1)[0]
        print(f"Model used (after fallback): {rec['model']}")
        print("(Fallback details in the log line above: opts=[model_fallback:gpt-4o->gpt-4o-mini])")

        # ------------------------------------------------------------------ #
        # 4. Cache — miss then hit
        # ------------------------------------------------------------------ #
        _section("4 / 5  Exact cache — miss then hit (OpenAI called only once)")

        mock_openai.chat.completions.create.reset_mock()
        mock_openai.chat.completions.create.return_value = _fake_response(
            "42.", prompt_tokens=50, completion_tokens=5
        )

        cache_client = DriftlockClient(
            api_key="demo",
            config=config,
            cache=CacheConfig(ttl_seconds=60, max_entries=100),
        )

        call_kwargs = dict(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 6 × 7?"}],
            temperature=0.0,
            _dl_endpoint="maths",
        )

        cache_client.chat.completions.create(**call_kwargs)   # MISS
        cache_client.chat.completions.create(**call_kwargs)   # HIT

        openai_call_count = mock_openai.chat.completions.create.call_count
        print(f"\nOpenAI API calls made  : {openai_call_count}  (expected 1)")
        print(f"Assertion passes       : {openai_call_count == 1}")

        recent = cache_client.recent_calls(limit=2)
        print(f"\nMost recent call  → cache_hit={recent[0]['cache_hit']}  "
              f"cost=${recent[0]['estimated_cost_usd']:.6f}  "
              f"saved={recent[0].get('tokens_saved_prompt', 0) + recent[0].get('tokens_saved_completion', 0)}tok")
        print(f"Previous call     → cache_hit={recent[1]['cache_hit']}  "
              f"cost=${recent[1]['estimated_cost_usd']:.6f}")

        cs = cache_client.cache_stats()
        print(f"\nCache stats: hits={cs['hits']}  misses={cs['misses']}  "
              f"hit_rate={cs['hit_rate']:.0%}  size={cs['size']}")

        # ------------------------------------------------------------------ #
        # 5. Context tags via driftlock.tag()
        # ------------------------------------------------------------------ #
        _section("5 / 5  Context tags — injected by middleware, visible in stored records")

        mock_openai.chat.completions.create.return_value = _fake_response("Sure!")

        tag_client = DriftlockClient(api_key="demo", config=config)

        with driftlock.tag(request_id="req_abc123", user_id="u_42", feature="search"):
            tag_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Find something for me."}],
                # No _dl_labels — tags come entirely from the context manager
            )

        rec = tag_client.recent_calls(limit=1)[0]
        print(f"\nLabels stored in SQLite: {json.dumps(rec['labels'], indent=2)}")

        # ------------------------------------------------------------------ #
        # Final summary
        # ------------------------------------------------------------------ #
        _section("Summary — aggregate stats across all demo calls")

        # Use the opt_client as the primary tracking store (it saw the most varied calls)
        all_calls = (
            client.stats()["calls"]
            + opt_client.stats()["calls"]
            + guard_client.stats()["calls"]
            + fallback_client.stats()["calls"]
            + cache_client.stats()["calls"]
            + tag_client.stats()["calls"]
        )
        print(f"\nTotal calls tracked across all demo clients: {all_calls}")

        print("\ncache_client.stats():")
        print(json.dumps(cache_client.stats(), indent=2))

        print("\ncache_client.cache_stats():")
        print(json.dumps(cache_client.cache_stats(), indent=2))

    print("\n" + "─" * 60)
    print("  Demo complete — all features exercised without a real API key.")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
