from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimization import OptimizationReport


@dataclass
class CallMetrics:
    """Metrics captured for a single LLM API call (real or cache hit)."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float | None

    # Provider and routing / labelling
    provider: str = "openai"
    endpoint: str | None = None
    labels: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str | None = None

    # Runtime warnings
    warnings: list[str] = field(default_factory=list)

    # Optimization pipeline report (present when optimizer is configured)
    optimization_report: OptimizationReport | None = None

    # Policy decisions (ordered list for transparency)
    policy_decisions: list[dict] = field(default_factory=list)

    # Rollout status counters (used for aggregated stats)
    optimization_enabled: bool = False
    optimization_shadow: bool = False
    sampled_out: bool = False
    quality_regression: bool = False

    # Prompt drift fingerprint (hash of template portion of the prompt)
    prompt_hash: str | None = None

    # Mission attribution (runtime financial guardrail for agents)
    mission_id: str | None = None
    parent_call_id: str | None = None

    # Cache fields (populated on a cache hit; zero/None otherwise)
    cache_hit: bool = False
    cache_key: str | None = None          # first 8 chars of the SHA-256 key (log prefix)
    tokens_saved_prompt: int = 0
    tokens_saved_completion: int = 0
    estimated_savings_usd: float | None = None

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "estimated_cost_usd": self.estimated_cost_usd,
            "labels": self.labels,
            "warnings": self.warnings,
            "request_id": self.request_id,
            "prompt_hash": self.prompt_hash,
            "cache_hit": self.cache_hit,
            "optimization_enabled": self.optimization_enabled,
            "optimization_shadow": self.optimization_shadow,
            "sampled_out": self.sampled_out,
            "quality_regression": self.quality_regression,
        }
        if self.cache_hit:
            d["cache_key"] = self.cache_key
            d["tokens_saved_prompt"] = self.tokens_saved_prompt
            d["tokens_saved_completion"] = self.tokens_saved_completion
            d["estimated_savings_usd"] = self.estimated_savings_usd
        if self.optimization_report is not None:
            d["optimization"] = self.optimization_report.to_dict()
        if self.policy_decisions:
            d["policy"] = list(self.policy_decisions)
        if self.mission_id is not None:
            d["mission_id"] = self.mission_id
        if self.parent_call_id is not None:
            d["parent_call_id"] = self.parent_call_id
        return d
