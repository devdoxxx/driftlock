from .alerts import (
    AlertChannel,
    LogAlertChannel,
    SlackAlertChannel,
    WebhookAlertChannel,
)
from .cache import CacheConfig
from .client import DriftlockClient
from .config import DriftlockConfig
from .context import tag
from .drift import detect_drift, hash_prompt
from .mission import (
    MissionBudgetExceededError,
    MissionContext,
    MissionSummary,
    mission,
)
from .optimization import BudgetExceededError, OptimizationConfig
from .policy import (
    BaseRule,
    CircuitOpenError,
    CostVelocityRule,
    ForecastBudgetRule,
    MaxCostPerRequestRule,
    MonthlyBudgetRule,
    PerUserBudgetRule,
    PolicyEngine,
    PolicyViolationError,
    RestrictModelRule,
    RuleDecision,
    TagBasedModelDowngradeRule,
    VelocityLimitRule,
)
from .providers import AnthropicProvider, NormalizedUsage, OpenAIProvider

__version__ = "0.5.0"
__all__ = [
    # Core clients
    "DriftlockClient",
    "DriftlockConfig",
    "OptimizationConfig",
    "BudgetExceededError",
    "CacheConfig",
    "tag",
    # Mission system (runtime financial guardrails for agents)
    "mission",
    "MissionContext",
    "MissionSummary",
    "MissionBudgetExceededError",
    # Policy engine
    "PolicyEngine",
    "PolicyViolationError",
    "CircuitOpenError",
    "RuleDecision",
    "BaseRule",
    "MaxCostPerRequestRule",
    "RestrictModelRule",
    "TagBasedModelDowngradeRule",
    "MonthlyBudgetRule",
    "PerUserBudgetRule",
    "VelocityLimitRule",
    "CostVelocityRule",
    "ForecastBudgetRule",
    # Alerts
    "AlertChannel",
    "WebhookAlertChannel",
    "SlackAlertChannel",
    "LogAlertChannel",
    # Providers
    "NormalizedUsage",
    "OpenAIProvider",
    "AnthropicProvider",
    # Drift
    "hash_prompt",
    "detect_drift",
]

# Anthropic client is opt-in (requires `pip install driftlock[anthropic]`)
try:
    from .anthropic_client import AnthropicDriftlockClient
    __all__.append("AnthropicDriftlockClient")
except ImportError:
    pass

# LangChain callback handler is import-safe even without langchain installed
# (`pip install driftlock[langchain]` only needed to actually wire it up).
try:
    from .integrations.langchain import DriftlockCallbackHandler
    __all__.append("DriftlockCallbackHandler")
except Exception:  # pragma: no cover - never block the package import
    pass
