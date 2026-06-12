"""
AnthropicDriftlockClient — transparent wrapper around the Anthropic Python SDK.

Usage::

    from driftlock import AnthropicDriftlockClient, DriftlockConfig

    client = AnthropicDriftlockClient(
        api_key="sk-ant-...",
        config=DriftlockConfig(),
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
        _dl_endpoint="my_function",
        _dl_labels={"env": "prod"},
    )

The Anthropic Messages API differences from OpenAI:
  - ``max_tokens`` is required (Anthropic enforces this)
  - ``system`` is a top-level kwarg, not a message role
  - Response uses ``usage.input_tokens`` / ``usage.output_tokens``
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

try:
    from anthropic import Anthropic, AsyncAnthropic
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "anthropic package is required for AnthropicDriftlockClient. "
        "Install it with: pip install 'driftlock[anthropic]'"
    ) from _e

from .cache import CacheConfig, ResponseCache, make_cache_key
from .config import DriftlockConfig
from .context import get_active_tags
from .drift import hash_prompt
from .logger import DriftlockLogger
from .metrics import CallMetrics
from .mission import (
    MissionSummary,
    build_mission_stats,
    current_mission,
    enforce_before_call,
    register_default_storage,
    resume_mission,
)
from .optimization import OptimizationConfig, OptimizationPipeline
from .policy import PolicyEngine, PolicyViolationError
from .pricing import estimate_cost
from .providers.anthropic_provider import AnthropicProvider
from .storage import NoopStorage, SQLiteStorage
from .client import _env_flag, _is_sampled_in, _sample_key_value


_PROVIDER = AnthropicProvider()


class _AnthropicMessagesWrapper:
    """Intercepts messages.create calls for Anthropic."""

    def __init__(self, sync_messages, async_messages, client: "AnthropicDriftlockClient") -> None:
        self._sync = sync_messages
        self._async = async_messages
        self._dl = client

    def create(self, *args, **kwargs) -> Any:
        # ------------------------------------------------------------------ #
        # 1. Strip Driftlock-only kwargs
        # ------------------------------------------------------------------ #
        endpoint: str | None = kwargs.pop("_dl_endpoint", None)
        labels: dict = kwargs.pop("_dl_labels", {})
        parent_call_id: str | None = kwargs.pop("_dl_parent_call_id", None)

        merged_labels: dict = {
            **self._dl._config.default_labels,
            **get_active_tags(),
            **labels,
        }

        enabled = _env_flag("DRIFTLOCK_ENABLED", True)
        track_only = _env_flag("DRIFTLOCK_TRACK_ONLY", False)

        if not enabled and not track_only:
            return self._sync.create(*args, **kwargs)

        # Mission attribution + mid-execution guardrail enforcement (walks the
        # full mission stack so an exhausted outer mission can halt the run).
        mission = current_mission()
        mission_id = mission.mission_id if mission is not None else None
        if mission is not None:
            enforce_before_call(kwargs)

        # ------------------------------------------------------------------ #
        # 2. Normalize messages for optimization pipeline
        #    Anthropic passes system as a top-level kwarg, not inside messages.
        #    We inject it as a synthetic system message for token counting /
        #    trimming, then strip it back out before the API call.
        # ------------------------------------------------------------------ #
        system_text: str | None = kwargs.pop("system", None)
        messages: list[dict] = list(kwargs.get("messages", []))
        if system_text:
            messages = [{"role": "system", "content": system_text}] + messages

        opt_report = None
        policy_decisions: list[dict] = []
        optimization_enabled = False
        optimization_shadow = False
        sampled_out = False

        optimizer = self._dl._optimizer if enabled else None
        cache = self._dl._cache if enabled else None

        if optimizer is not None:
            cfg = self._dl._optimization_config
            optimization_shadow = bool(cfg.shadow_mode)
            sample_rate = max(0.0, min(1.0, cfg.sample_rate))
            sample_value = _sample_key_value(cfg.sample_key, merged_labels, kwargs)
            sampled_in = _is_sampled_in(sample_value, sample_rate)
            sampled_out = not sampled_in
            apply = sampled_in and not optimization_shadow

            model, messages, kwargs, opt_report = optimizer.process(
                model=kwargs.get("model", "unknown"),
                messages=messages,
                kwargs=kwargs,
                apply=apply,
                shadow_mode=optimization_shadow,
            )
            if opt_report and sampled_out:
                opt_report.bypassed_reason = "sampled_out"

            kwargs["model"] = model
            optimization_enabled = not sampled_out

        # Extract system message back out (Anthropic requires it as a kwarg)
        non_system = [m for m in messages if m.get("role") != "system"]
        remaining_system = [m for m in messages if m.get("role") == "system"]
        kwargs["messages"] = non_system
        if remaining_system:
            # Re-attach system text (concatenate if multiple were kept)
            kwargs["system"] = "\n\n".join(
                m["content"] for m in remaining_system if isinstance(m.get("content"), str)
            )
        elif system_text and not (optimizer and opt_report):
            # No optimizer ran; restore the original system kwarg
            kwargs["system"] = system_text

        # ------------------------------------------------------------------ #
        # 3. Policy evaluation
        # ------------------------------------------------------------------ #
        if self._dl._policy is not None:
            ctx = {
                "model": kwargs.get("model", "unknown"),
                "messages": kwargs.get("messages", []),
                "kwargs": kwargs,
                "labels": merged_labels,
                "optimization_report": opt_report,
                "storage": self._dl._storage,
            }
            for rule_name, decision in self._dl._policy.evaluate(ctx):
                policy_decisions.append(
                    {
                        "rule": rule_name,
                        "allow": decision.allow,
                        "action": decision.action,
                        "metadata": decision.metadata,
                    }
                )
                if not decision.allow or decision.action == "block":
                    raise PolicyViolationError(rule_name, decision)
                if decision.action in {"downgrade", "fallback"}:
                    target = (
                        decision.metadata.get("downgrade_to")
                        or decision.metadata.get("fallback_to")
                        or decision.metadata.get("fallback_model")
                    )
                    if target:
                        kwargs["model"] = target

        # ------------------------------------------------------------------ #
        # 4. Cache lookup (key computed after optimization)
        # ------------------------------------------------------------------ #
        cache_key: str | None = None
        if cache is not None and not kwargs.get("stream", False):
            cache_key = make_cache_key(
                model=kwargs.get("model", "unknown"),
                messages=kwargs.get("messages", []),
                kwargs=kwargs,
            )
            t0 = time.perf_counter()
            entry = cache.get(cache_key)
            latency_ms = (time.perf_counter() - t0) * 1000

            if entry is not None:
                model_name = kwargs.get("model", "unknown")
                savings_usd = estimate_cost(
                    model_name, entry.prompt_tokens, entry.completion_tokens
                )
                metrics = CallMetrics(
                    provider="anthropic",
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=latency_ms,
                    estimated_cost_usd=0.0,
                    endpoint=endpoint,
                    labels=merged_labels,
                    request_id=str(uuid.uuid4()),
                    optimization_report=opt_report,
                    policy_decisions=policy_decisions,
                    cache_hit=True,
                    cache_key=cache_key[:8],
                    tokens_saved_prompt=entry.prompt_tokens,
                    tokens_saved_completion=entry.completion_tokens,
                    estimated_savings_usd=savings_usd,
                    optimization_enabled=optimization_enabled,
                    optimization_shadow=optimization_shadow,
                    sampled_out=sampled_out,
                    mission_id=mission_id,
                    parent_call_id=parent_call_id,
                )
                self._dl._logger.log_call(metrics)
                self._dl._storage.save(metrics)
                if mission is not None:
                    mission._record_call(metrics, self._dl._storage)
                return entry.response

        # ------------------------------------------------------------------ #
        # 5. Cache miss — call the real API
        # ------------------------------------------------------------------ #
        start = time.perf_counter()
        response = self._sync.create(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = _PROVIDER.normalize_response(response)

        if cache is not None and cache_key is not None:
            cache.put(cache_key, response, usage.prompt_tokens, usage.completion_tokens)

        # ------------------------------------------------------------------ #
        # 6. Metrics, warnings, logging, storage
        # ------------------------------------------------------------------ #
        cost = estimate_cost(usage.model, usage.prompt_tokens, usage.completion_tokens)
        cfg = self._dl._config
        warnings: list[str] = []

        if usage.prompt_tokens > cfg.prompt_token_warning_threshold:
            warnings.append(
                f"Prompt is large: {usage.prompt_tokens} tokens "
                f"(threshold: {cfg.prompt_token_warning_threshold})"
            )
        if cfg.cost_warning_threshold and cost and cost > cfg.cost_warning_threshold:
            warnings.append(
                f"Call cost ${cost:.6f} exceeds warning threshold "
                f"${cfg.cost_warning_threshold:.6f}"
            )

        p_hash = hash_prompt(kwargs.get("messages", []), kwargs.get("system"))

        metrics = CallMetrics(
            provider="anthropic",
            model=usage.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.prompt_tokens + usage.completion_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            endpoint=endpoint,
            labels=merged_labels,
            request_id=str(uuid.uuid4()),
            warnings=warnings,
            prompt_hash=p_hash,
            optimization_report=opt_report,
            policy_decisions=policy_decisions,
            optimization_enabled=optimization_enabled,
            optimization_shadow=optimization_shadow,
            sampled_out=sampled_out,
            mission_id=mission_id,
            parent_call_id=parent_call_id,
        )

        self._dl._logger.log_call(metrics)
        self._dl._storage.save(metrics)
        if mission is not None:
            mission._record_call(metrics, self._dl._storage)
        return response

    async def acreate(self, *args, **kwargs) -> Any:
        """Async version of create().  Requires AsyncAnthropic under the hood."""
        import asyncio

        # ------------------------------------------------------------------ #
        # 1. Strip Driftlock-only kwargs (sync — trivial)
        # ------------------------------------------------------------------ #
        endpoint: str | None = kwargs.pop("_dl_endpoint", None)
        labels: dict = kwargs.pop("_dl_labels", {})
        parent_call_id: str | None = kwargs.pop("_dl_parent_call_id", None)

        merged_labels: dict = {
            **self._dl._config.default_labels,
            **get_active_tags(),
            **labels,
        }

        enabled = _env_flag("DRIFTLOCK_ENABLED", True)
        track_only = _env_flag("DRIFTLOCK_TRACK_ONLY", False)

        if not enabled and not track_only:
            return await self._async.create(*args, **kwargs)

        # Mission attribution + mid-execution guardrail enforcement (walks the
        # full mission stack so an exhausted outer mission can halt the run).
        mission = current_mission()
        mission_id = mission.mission_id if mission is not None else None
        if mission is not None:
            enforce_before_call(kwargs)

        system_text: str | None = kwargs.pop("system", None)
        messages: list[dict] = list(kwargs.get("messages", []))
        if system_text:
            messages = [{"role": "system", "content": system_text}] + messages

        opt_report = None
        policy_decisions: list[dict] = []
        optimization_enabled = False
        optimization_shadow = False
        sampled_out = False

        optimizer = self._dl._optimizer if enabled else None
        cache = self._dl._cache if enabled else None

        if optimizer is not None:
            cfg = self._dl._optimization_config
            optimization_shadow = bool(cfg.shadow_mode)
            sample_rate = max(0.0, min(1.0, cfg.sample_rate))
            sample_value = _sample_key_value(cfg.sample_key, merged_labels, kwargs)
            sampled_in = _is_sampled_in(sample_value, sample_rate)
            sampled_out = not sampled_in
            apply = sampled_in and not optimization_shadow

            model, messages, kwargs, opt_report = optimizer.process(
                model=kwargs.get("model", "unknown"),
                messages=messages,
                kwargs=kwargs,
                apply=apply,
                shadow_mode=optimization_shadow,
            )
            if opt_report and sampled_out:
                opt_report.bypassed_reason = "sampled_out"

            kwargs["model"] = model
            optimization_enabled = not sampled_out

        non_system = [m for m in messages if m.get("role") != "system"]
        remaining_system = [m for m in messages if m.get("role") == "system"]
        kwargs["messages"] = non_system
        if remaining_system:
            kwargs["system"] = "\n\n".join(
                m["content"] for m in remaining_system if isinstance(m.get("content"), str)
            )
        elif system_text and not (optimizer and opt_report):
            kwargs["system"] = system_text

        if self._dl._policy is not None:
            ctx = {
                "model": kwargs.get("model", "unknown"),
                "messages": kwargs.get("messages", []),
                "kwargs": kwargs,
                "labels": merged_labels,
                "optimization_report": opt_report,
                "storage": self._dl._storage,
            }
            for rule_name, decision in self._dl._policy.evaluate(ctx):
                policy_decisions.append(
                    {
                        "rule": rule_name,
                        "allow": decision.allow,
                        "action": decision.action,
                        "metadata": decision.metadata,
                    }
                )
                if not decision.allow or decision.action == "block":
                    raise PolicyViolationError(rule_name, decision)
                if decision.action in {"downgrade", "fallback"}:
                    target = (
                        decision.metadata.get("downgrade_to")
                        or decision.metadata.get("fallback_to")
                        or decision.metadata.get("fallback_model")
                    )
                    if target:
                        kwargs["model"] = target

        cache_key: str | None = None
        if cache is not None and not kwargs.get("stream", False):
            cache_key = make_cache_key(
                model=kwargs.get("model", "unknown"),
                messages=kwargs.get("messages", []),
                kwargs=kwargs,
            )
            entry = cache.get(cache_key)
            if entry is not None:
                model_name = kwargs.get("model", "unknown")
                savings_usd = estimate_cost(
                    model_name, entry.prompt_tokens, entry.completion_tokens
                )
                metrics = CallMetrics(
                    provider="anthropic",
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=0.0,
                    estimated_cost_usd=0.0,
                    endpoint=endpoint,
                    labels=merged_labels,
                    request_id=str(uuid.uuid4()),
                    cache_hit=True,
                    cache_key=cache_key[:8],
                    tokens_saved_prompt=entry.prompt_tokens,
                    tokens_saved_completion=entry.completion_tokens,
                    estimated_savings_usd=savings_usd,
                    optimization_enabled=optimization_enabled,
                    optimization_shadow=optimization_shadow,
                    sampled_out=sampled_out,
                    mission_id=mission_id,
                    parent_call_id=parent_call_id,
                )
                self._dl._logger.log_call(metrics)
                await asyncio.to_thread(self._dl._storage.save, metrics)
                if mission is not None:
                    await mission._arecord_call(metrics, self._dl._storage)
                return entry.response

        start = time.perf_counter()
        response = await self._async.create(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = _PROVIDER.normalize_response(response)

        if cache is not None and cache_key is not None:
            cache.put(cache_key, response, usage.prompt_tokens, usage.completion_tokens)

        cost = estimate_cost(usage.model, usage.prompt_tokens, usage.completion_tokens)
        cfg = self._dl._config
        warnings: list[str] = []

        if usage.prompt_tokens > cfg.prompt_token_warning_threshold:
            warnings.append(
                f"Prompt is large: {usage.prompt_tokens} tokens "
                f"(threshold: {cfg.prompt_token_warning_threshold})"
            )
        if cfg.cost_warning_threshold and cost and cost > cfg.cost_warning_threshold:
            warnings.append(
                f"Call cost ${cost:.6f} exceeds warning threshold "
                f"${cfg.cost_warning_threshold:.6f}"
            )

        p_hash = hash_prompt(kwargs.get("messages", []), kwargs.get("system"))

        metrics = CallMetrics(
            provider="anthropic",
            model=usage.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.prompt_tokens + usage.completion_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            endpoint=endpoint,
            labels=merged_labels,
            request_id=str(uuid.uuid4()),
            warnings=warnings,
            prompt_hash=p_hash,
            optimization_report=opt_report,
            policy_decisions=policy_decisions,
            optimization_enabled=optimization_enabled,
            optimization_shadow=optimization_shadow,
            sampled_out=sampled_out,
            mission_id=mission_id,
            parent_call_id=parent_call_id,
        )

        self._dl._logger.log_call(metrics)
        await asyncio.to_thread(self._dl._storage.save, metrics)
        if mission is not None:
            await mission._arecord_call(metrics, self._dl._storage)
        return response


class _AnthropicMessagesClientWrapper:
    def __init__(
        self,
        sync_client: Anthropic,
        async_client: AsyncAnthropic,
        dl_client: "AnthropicDriftlockClient",
    ) -> None:
        self.messages = _AnthropicMessagesWrapper(
            sync_client.messages,
            async_client.messages,
            dl_client,
        )


class AnthropicDriftlockClient:
    """
    Drop-in wrapper around anthropic.Anthropic.

    Adds token tracking, cost estimation, latency measurement, structured
    logging, an optional optimization pipeline, cache, and policy engine.

    All kwargs not listed below are forwarded to anthropic.Anthropic.
    """

    def __init__(
        self,
        *,
        config: DriftlockConfig | None = None,
        optimization: OptimizationConfig | None = None,
        cache: CacheConfig | None = None,
        policy: PolicyEngine | None = None,
        **anthropic_kwargs: Any,
    ) -> None:
        self._config = config or DriftlockConfig()
        self._sync_anthropic = Anthropic(**anthropic_kwargs)
        self._async_anthropic = AsyncAnthropic(**anthropic_kwargs)
        self._logger = DriftlockLogger(
            log_json=self._config.log_json,
            log_level=self._config.log_level,
        )
        self._optimization_config = optimization or OptimizationConfig()
        self._optimizer = (
            OptimizationPipeline(self._optimization_config) if optimization else None
        )
        self._cache = ResponseCache(cache) if (cache and cache.enabled) else None
        self._policy = policy

        if self._config.storage_backend == "sqlite":
            self._storage = SQLiteStorage(self._config.db_path)
        else:
            self._storage = NoopStorage()
        register_default_storage(self._storage)

        self.messages = _AnthropicMessagesClientWrapper(
            self._sync_anthropic, self._async_anthropic, self
        ).messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sync_anthropic, name)

    def stats(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        since: str | None = None,
    ) -> dict:
        """Return aggregated metrics from local storage."""
        return self._storage.aggregate(
            endpoint=endpoint, model=model, provider="anthropic", since=since
        )

    def recent_calls(self, limit: int = 20) -> list[dict]:
        """Return the N most recent tracked calls."""
        return self._storage.recent(limit=limit)

    def mission_stats(self, mission_id: str) -> dict:
        """
        Return an aggregate report for a mission: total spend, call count,
        the parent/child call graph, per-model distribution, and the list of
        intervention events that fired during the run.
        """
        return build_mission_stats(self._storage, mission_id)

    def missions(self, limit: int = 20, since: str | None = None) -> list[dict]:
        """Return recent missions with spend, call count, and final status."""
        return self._storage.list_missions(limit=limit, since=since)

    def resume_mission(self, mission_id: str) -> MissionSummary | None:
        """Return a read-only :class:`MissionSummary` for a past mission."""
        return resume_mission(self._storage, mission_id)

    def cache_stats(self) -> dict:
        """Return live cache hit/miss stats."""
        if self._cache is None:
            return {"enabled": False}
        return {"enabled": True, **self._cache.stats()}
