"""
DriftlockClient — a transparent wrapper around multiple different LLM providers (currently just OpenAI) that adds:

Usage::

    from driftlock import DriftlockClient, DriftlockConfig, OptimizationConfig, CacheConfig

    client = DriftlockClient(
        api_key="sk-...",
        optimization=OptimizationConfig(max_prompt_tokens=3000),
        cache=CacheConfig(ttl_seconds=600),
    )

    # Optionally set ambient tags for a whole block (e.g. from middleware):
    with driftlock.tag(request_id="req_123", user_id="u_42"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
            _dl_endpoint="my_function",   # per-call label
            _dl_labels={"env": "prod"},   # per-call labels
        )
"""

import hashlib
import os
import time
import uuid
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from .alerts import ALERT_COST_WARNING, ALERT_POLICY_BLOCK, fire_alert
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
from .providers.openai_provider import OpenAIProvider
from .storage import NoopStorage, SQLiteStorage
from .streaming import StreamingInterceptor
from .tokenizer import count_messages_tokens

_PROVIDER = OpenAIProvider()


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


def _sample_key_value(sample_key: str, labels: dict, kwargs: dict) -> str:
    if sample_key in labels:
        return str(labels[sample_key])
    if sample_key in kwargs:
        return str(kwargs[sample_key])
    return "unknown"


def _is_sampled_in(sample_value: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(sample_value.encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < rate


class _ChatCompletionsWrapper:
    """Intercepts chat.completions.create / acreate calls."""

    def __init__(
        self,
        sync_completions,
        async_completions,
        client: "DriftlockClient",
    ) -> None:
        self._sync = sync_completions
        self._async = async_completions
        self._dl = client

    def create(self, *args, **kwargs) -> ChatCompletion:
        # ------------------------------------------------------------------ #
        # 1. Strip Driftlock-only kwargs
        # ------------------------------------------------------------------ #
        endpoint: str | None = kwargs.pop("_dl_endpoint", None)
        labels: dict = kwargs.pop("_dl_labels", {})
        parent_call_id: str | None = kwargs.pop("_dl_parent_call_id", None)

        # Merge precedence: default_labels < context tags < per-call labels
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
        # 2. Optimization pipeline (trimming → output cap → budget guardrail)
        # ------------------------------------------------------------------ #
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
                messages=kwargs.get("messages", []),
                kwargs=kwargs,
                apply=apply,
                shadow_mode=optimization_shadow,
            )
            if opt_report and sampled_out:
                opt_report.bypassed_reason = "sampled_out"

            kwargs["model"] = model
            kwargs["messages"] = messages
            optimization_enabled = not sampled_out

        # ------------------------------------------------------------------ #
        # 3. Policy evaluation (after optimization, before cache)
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
                    fire_alert(
                        self._dl._config.alert_channels,
                        ALERT_POLICY_BLOCK,
                        {"rule": rule_name, "model": kwargs.get("model"), **decision.metadata},
                    )
                    raise PolicyViolationError(rule_name, decision)
                if decision.action in {"downgrade", "fallback"}:
                    target = (
                        decision.metadata.get("downgrade_to")
                        or decision.metadata.get("fallback_to")
                        or decision.metadata.get("fallback_model")
                    )
                    if target:
                        kwargs["model"] = target
                        ctx["model"] = target

        # ------------------------------------------------------------------ #
        # 4. Cache lookup (key computed AFTER optimization)
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
                    provider="openai",
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

        # Streaming: wrap response for deferred token counting
        if kwargs.get("stream", False):
            pre_tokens = count_messages_tokens(kwargs.get("messages", []), kwargs.get("model", "gpt-4o"))
            raw_stream = self._sync.create(*args, **kwargs)
            return StreamingInterceptor(
                stream=raw_stream,
                model=kwargs.get("model", "unknown"),
                messages=kwargs.get("messages", []),
                pre_call_prompt_tokens=pre_tokens,
                start_time=start,
                endpoint=endpoint,
                labels=merged_labels,
                storage=self._dl._storage,
                logger=self._dl._logger,
                config=self._dl._config,
                optimization_report=opt_report,
                policy_decisions=policy_decisions,
                mission_id=mission_id,
                parent_call_id=parent_call_id,
                mission=mission,
            )

        response: ChatCompletion = self._sync.create(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        norm = _PROVIDER.normalize_response(response)

        if cache is not None and cache_key is not None:
            cache.put(cache_key, response, norm.prompt_tokens, norm.completion_tokens)

        # ------------------------------------------------------------------ #
        # 6. Metrics, warnings, logging, storage
        # ------------------------------------------------------------------ #
        cost = estimate_cost(norm.model, norm.prompt_tokens, norm.completion_tokens)
        cfg = self._dl._config
        warnings: list[str] = []

        if norm.prompt_tokens > cfg.prompt_token_warning_threshold:
            warnings.append(
                f"Prompt is large: {norm.prompt_tokens} tokens "
                f"(threshold: {cfg.prompt_token_warning_threshold})"
            )
        if cfg.cost_warning_threshold and cost and cost > cfg.cost_warning_threshold:
            warnings.append(
                f"Call cost ${cost:.6f} exceeds warning threshold "
                f"${cfg.cost_warning_threshold:.6f}"
            )
            fire_alert(
                cfg.alert_channels,
                ALERT_COST_WARNING,
                {"model": norm.model, "cost_usd": cost, "threshold_usd": cfg.cost_warning_threshold},
            )

        p_hash = hash_prompt(kwargs.get("messages", []))

        metrics = CallMetrics(
            provider="openai",
            model=norm.model,
            prompt_tokens=norm.prompt_tokens,
            completion_tokens=norm.completion_tokens,
            total_tokens=norm.prompt_tokens + norm.completion_tokens,
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

    async def acreate(self, *args, **kwargs) -> ChatCompletion:
        """Async version of create().  Uses AsyncOpenAI under the hood."""
        import asyncio

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
            return await self._async.create(*args, **kwargs)

        # Mission attribution + mid-execution guardrail enforcement (walks the
        # full mission stack so an exhausted outer mission can halt the run).
        mission = current_mission()
        mission_id = mission.mission_id if mission is not None else None
        if mission is not None:
            enforce_before_call(kwargs)

        # ------------------------------------------------------------------ #
        # 2. Optimization pipeline
        # ------------------------------------------------------------------ #
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
                messages=kwargs.get("messages", []),
                kwargs=kwargs,
                apply=apply,
                shadow_mode=optimization_shadow,
            )
            if opt_report and sampled_out:
                opt_report.bypassed_reason = "sampled_out"

            kwargs["model"] = model
            kwargs["messages"] = messages
            optimization_enabled = not sampled_out

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
                    fire_alert(
                        self._dl._config.alert_channels,
                        ALERT_POLICY_BLOCK,
                        {"rule": rule_name, "model": kwargs.get("model"), **decision.metadata},
                    )
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
        # 4. Cache lookup
        # ------------------------------------------------------------------ #
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
                    provider="openai",
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

        # ------------------------------------------------------------------ #
        # 5. Cache miss — async API call
        # ------------------------------------------------------------------ #
        start = time.perf_counter()
        response: ChatCompletion = await self._async.create(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        norm = _PROVIDER.normalize_response(response)

        if cache is not None and cache_key is not None:
            cache.put(cache_key, response, norm.prompt_tokens, norm.completion_tokens)

        # ------------------------------------------------------------------ #
        # 6. Metrics, logging, storage
        # ------------------------------------------------------------------ #
        cost = estimate_cost(norm.model, norm.prompt_tokens, norm.completion_tokens)
        cfg = self._dl._config
        warnings: list[str] = []

        if norm.prompt_tokens > cfg.prompt_token_warning_threshold:
            warnings.append(
                f"Prompt is large: {norm.prompt_tokens} tokens "
                f"(threshold: {cfg.prompt_token_warning_threshold})"
            )
        if cfg.cost_warning_threshold and cost and cost > cfg.cost_warning_threshold:
            warnings.append(
                f"Call cost ${cost:.6f} exceeds warning threshold "
                f"${cfg.cost_warning_threshold:.6f}"
            )

        p_hash = hash_prompt(kwargs.get("messages", []))

        metrics = CallMetrics(
            provider="openai",
            model=norm.model,
            prompt_tokens=norm.prompt_tokens,
            completion_tokens=norm.completion_tokens,
            total_tokens=norm.prompt_tokens + norm.completion_tokens,
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


class _ChatWrapper:
    def __init__(
        self,
        sync_chat,
        async_chat,
        client: "DriftlockClient",
    ) -> None:
        self.completions = _ChatCompletionsWrapper(
            sync_chat.completions,
            async_chat.completions,
            client,
        )


class DriftlockClient:
    """
    Drop-in wrapper around openai.OpenAI.

    Adds token tracking, cost estimation, latency measurement, structured
    logging, an optional pre-call optimization pipeline, and an optional
    exact in-memory response cache.

    All kwargs not listed below are forwarded to openai.OpenAI.
    """

    def __init__(
        self,
        *,
        config: DriftlockConfig | None = None,
        optimization: OptimizationConfig | None = None,
        cache: CacheConfig | None = None,
        policy: PolicyEngine | None = None,
        **openai_kwargs: Any,
    ) -> None:
        self._config = config or DriftlockConfig()
        self._openai = OpenAI(**openai_kwargs)
        self._async_openai = AsyncOpenAI(**openai_kwargs)
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

        self.chat = _ChatWrapper(self._openai.chat, self._async_openai.chat, self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._openai, name)

    def stats(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        since: str | None = None,
    ) -> dict:
        """Return aggregated metrics from local storage (includes cache savings)."""
        return self._storage.aggregate(endpoint=endpoint, model=model, since=since)

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
        """
        Return a read-only :class:`MissionSummary` for a past mission — useful
        for post-run analysis without rehydrating a live ``MissionContext``.
        """
        return resume_mission(self._storage, mission_id)

    def cache_stats(self) -> dict:
        """Return live cache hit/miss stats (in-memory only, not persisted)."""
        if self._cache is None:
            return {"enabled": False}
        return {"enabled": True, **self._cache.stats()}

    def forecast(self, lookback_days: int = 7) -> dict:
        """
        Extrapolate current daily spend rate to a full 30-day month.

        Returns:
          - daily_avg_usd: average spend per day over the lookback window
          - projected_monthly_usd: daily_avg * 30
          - lookback_days: number of days used for the estimate
          - data_points: number of days with recorded data
        """
        trend = self._storage.daily_cost_trend(lookback_days=lookback_days)
        if not trend:
            return {
                "daily_avg_usd": 0.0,
                "projected_monthly_usd": 0.0,
                "lookback_days": lookback_days,
                "data_points": 0,
            }
        total = sum(d["cost_usd"] for d in trend)
        daily_avg = total / lookback_days
        return {
            "daily_avg_usd": round(daily_avg, 6),
            "projected_monthly_usd": round(daily_avg * 30, 4),
            "lookback_days": lookback_days,
            "data_points": len(trend),
        }

    def prompt_drift(self, endpoint: str, limit: int = 30) -> list[dict]:
        """
        Return a list of prompt hash change events for an endpoint.
        Each entry has: timestamp, old_hash, new_hash, prompt_tokens.
        """
        from .drift import detect_drift
        history = self._storage.prompt_hash_history(endpoint, limit=limit)
        return detect_drift(history)
