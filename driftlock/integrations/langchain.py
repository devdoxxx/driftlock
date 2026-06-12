"""
LangChain compatibility shim.

``DriftlockCallbackHandler`` lets a LangChain agent participate in the Driftlock
mission system without using ``DriftlockClient`` directly. Wrap the run in a
mission as usual and attach the handler::

    import driftlock
    from driftlock.integrations import DriftlockCallbackHandler
    from langchain_openai import ChatOpenAI

    handler = DriftlockCallbackHandler(client=dl_client)  # client persists metrics
    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

    with driftlock.mission("lc_agent", budget_usd=0.50, on_exceed="kill"):
        agent.invoke(...)   # every LLM call is attributed to the mission

The handler is import-safe even when LangChain is not installed (``langchain`` is
an optional extra), so it can be unit-tested without the dependency.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ..metrics import CallMetrics
from ..mission import current_mission
from ..pricing import estimate_cost

_log = logging.getLogger("driftlock")

# LangChain's BaseCallbackHandler if available, else a minimal stand-in so the
# handler stays importable/testable without the optional dependency.
try:  # pragma: no cover - exercised differently depending on install
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover
    try:
        from langchain.callbacks.base import BaseCallbackHandler
    except Exception:
        class BaseCallbackHandler:  # type: ignore[no-redef]
            """Fallback base used when LangChain is not installed."""


class DriftlockCallbackHandler(BaseCallbackHandler):
    """
    Bridge LangChain LLM lifecycle callbacks into the active Driftlock mission.

    Args:
        client: Optional ``DriftlockClient`` / ``AnthropicDriftlockClient`` whose
            storage should persist the synthesized call records (so
            ``mission_stats`` and the CLI see LangChain calls). If omitted, the
            mission's spend still updates in memory but nothing is persisted.
        endpoint: Optional label applied to recorded calls.
        default_model: Model name to fall back to when LangChain's ``llm_output``
            does not report one.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        endpoint: str | None = "langchain",
        default_model: str = "unknown",
    ) -> None:
        self._storage = getattr(client, "_storage", None)
        self._endpoint = endpoint
        self._default_model = default_model
        self.mission_id: str | None = None

    # ------------------------------------------------------------------ #
    # LangChain callback API
    # ------------------------------------------------------------------ #

    def on_llm_start(
        self,
        serialized: dict | None,
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Enforce the active mission's guardrail before the call goes out."""
        mission = current_mission()
        if mission is None:
            return
        self.mission_id = mission.mission_id
        # Link traces: stamp mission_id into LangChain's metadata if present.
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            metadata.setdefault("driftlock_mission_id", mission.mission_id)
        # Enforce kill/downgrade state. Raises MissionBudgetExceededError when
        # the mission is killed, halting the chain mid-run.
        mission._before_call({})

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Record token usage from a completed LLM call into the mission."""
        mission = current_mission()
        if mission is None:
            return
        model, prompt_tokens, completion_tokens = _extract_usage(
            response, self._default_model
        )
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")
        metrics = CallMetrics(
            provider="langchain",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=0.0,
            estimated_cost_usd=cost,
            endpoint=self._endpoint,
            labels={"mission_id": mission.mission_id},
            request_id=str(run_id) if run_id else str(uuid.uuid4()),
            parent_call_id=str(parent_run_id) if parent_run_id else None,
            mission_id=mission.mission_id,
        )
        if self._storage is not None:
            try:
                self._storage.save(metrics)
            except Exception:  # pragma: no cover - persistence is best-effort
                pass
        mission._record_call(metrics, self._storage)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Log the error; never corrupt mission spend on a failed call."""
        _log.warning("driftlock: LangChain LLM error (mission spend untouched): %s", error)


def _extract_usage(response: Any, default_model: str) -> tuple[str, int, int]:
    """Pull (model, prompt_tokens, completion_tokens) from a LangChain LLMResult."""
    llm_output = getattr(response, "llm_output", None) or {}
    if not isinstance(llm_output, dict):
        llm_output = {}
    usage = (
        llm_output.get("token_usage")
        or llm_output.get("usage")
        or {}
    )
    model = llm_output.get("model_name") or llm_output.get("model") or default_model
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(
        usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    )
    return model, prompt_tokens, completion_tokens
