"""
Streaming response interceptor.

Wraps an OpenAI streaming response (``stream=True``) to:
  - Count prompt and completion tokens as chunks arrive
  - Record a CallMetrics entry when the stream closes
  - Fire cost warnings if configured

Token counting strategy:
  - prompt_tokens: taken from the first chunk's ``usage`` field when the API
    provides it (stream_options={"include_usage": True}).  Falls back to a
    pre-call estimate via count_messages_tokens.
  - completion_tokens: accumulated from each chunk's delta content length,
    then finalized from the terminal ``usage`` chunk if available.

Usage:
    response = client._openai.chat.completions.create(
        model=..., messages=..., stream=True,
        stream_options={"include_usage": True},
    )
    wrapped = StreamingInterceptor(
        stream=response,
        model=model,
        messages=messages,
        pre_call_prompt_tokens=estimate,
        ...
    )
    for chunk in wrapped:   # or ``async for chunk in wrapped``
        yield chunk
    # metrics are recorded automatically when the stream ends
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Generator, Iterator

from .alerts import ALERT_COST_WARNING, fire_alert
from .drift import hash_prompt
from .logger import DriftlockLogger
from .metrics import CallMetrics
from .pricing import estimate_cost
from .tokenizer import count_tokens


class StreamingInterceptor:
    """
    Wraps a synchronous OpenAI streaming response to track tokens and cost.

    Iterate over this object exactly as you would the raw stream.
    Metrics are persisted when the generator is exhausted or closed.
    """

    def __init__(
        self,
        stream: Any,
        *,
        model: str,
        messages: list[dict],
        pre_call_prompt_tokens: int,
        start_time: float,
        endpoint: str | None,
        labels: dict,
        storage: Any,
        logger: DriftlockLogger,
        config: Any,
        optimization_report: Any = None,
        policy_decisions: list[dict] | None = None,
        mission_id: str | None = None,
        parent_call_id: str | None = None,
        mission: Any = None,
    ) -> None:
        self._stream = stream
        self._model = model
        self._messages = messages
        self._pre_prompt_tokens = pre_call_prompt_tokens
        self._start = start_time
        self._endpoint = endpoint
        self._labels = labels
        self._storage = storage
        self._logger = logger
        self._config = config
        self._opt_report = optimization_report
        self._policy_decisions = policy_decisions or []
        self._mission_id = mission_id
        self._parent_call_id = parent_call_id
        self._mission = mission

        self._completion_chars = 0
        self._prompt_tokens_final: int | None = None
        self._completion_tokens_final: int | None = None
        self._recorded = False

    def __iter__(self) -> Iterator[Any]:
        try:
            for chunk in self._stream:
                self._process_chunk(chunk)
                yield chunk
        finally:
            self._record()

    def _process_chunk(self, chunk: Any) -> None:
        # OpenAI streams usage in the final chunk when stream_options={"include_usage": True}
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._prompt_tokens_final = getattr(usage, "prompt_tokens", None)
            self._completion_tokens_final = getattr(usage, "completion_tokens", None)

        # Accumulate completion chars as fallback counter
        choices = getattr(chunk, "choices", []) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if delta:
                content = getattr(delta, "content", None) or ""
                self._completion_chars += len(content)

    def _record(self) -> None:
        if self._recorded:
            return
        self._recorded = True

        latency_ms = (time.perf_counter() - self._start) * 1000
        prompt_tokens = self._prompt_tokens_final or self._pre_prompt_tokens
        completion_tokens = (
            self._completion_tokens_final
            or max(1, self._completion_chars // 4)
        )

        cost = estimate_cost(self._model, prompt_tokens, completion_tokens)
        cfg = self._config
        warnings: list[str] = []

        if prompt_tokens > cfg.prompt_token_warning_threshold:
            warnings.append(
                f"Prompt is large: {prompt_tokens} tokens "
                f"(threshold: {cfg.prompt_token_warning_threshold})"
            )
        if cfg.cost_warning_threshold and cost and cost > cfg.cost_warning_threshold:
            warnings.append(
                f"Streaming call cost ${cost:.6f} exceeds warning threshold "
                f"${cfg.cost_warning_threshold:.6f}"
            )
            fire_alert(
                cfg.alert_channels,
                ALERT_COST_WARNING,
                {"model": self._model, "cost_usd": cost, "threshold_usd": cfg.cost_warning_threshold},
            )

        p_hash = hash_prompt(self._messages)

        metrics = CallMetrics(
            provider="openai",
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            endpoint=self._endpoint,
            labels=self._labels,
            request_id=str(uuid.uuid4()),
            warnings=warnings,
            prompt_hash=p_hash,
            optimization_report=self._opt_report,
            policy_decisions=self._policy_decisions,
            mission_id=self._mission_id,
            parent_call_id=self._parent_call_id,
        )

        self._logger.log_call(metrics)
        self._storage.save(metrics)
        if self._mission is not None:
            self._mission._record_call(metrics, self._storage)
