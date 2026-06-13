# Contributing to Driftlock

Thanks for your interest in improving Driftlock. This page covers everything you
need to get productive in a few minutes.

## Dev environment

```bash
git clone https://github.com/maddox-214/driftlock && cd driftlock
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # editable install + pytest, ruff, etc.
```

Driftlock targets **Python 3.11+** and has one runtime dependency (`openai`).
Everything else — Anthropic, LangChain, LangGraph, FastAPI — is an optional
extra, so keep new hard dependencies out of the core.

## Running the tests

```bash
pytest                       # full suite (currently 235 tests, runs in ~3s)
pytest tests/test_mission.py # one file
pytest -k downgrade          # by keyword
```

No tests hit the network — the suite mocks the provider SDKs and uses in-memory
or temp-file SQLite. A change is not done until `pytest` is green.

## Linting

```bash
ruff check driftlock/ tests/ examples/
```

Keep new code ruff-clean. Match the surrounding style (line length 100, type
hints, module-level docstrings).

## Adding a new provider adapter

Providers normalize a vendor's response into `NormalizedUsage`. Use the existing
adapters as the pattern:

- [`driftlock/providers/base.py`](driftlock/providers/base.py) — the
  `NormalizedUsage` shape and base class.
- [`driftlock/providers/openai_provider.py`](driftlock/providers/openai_provider.py)
  and [`anthropic_provider.py`](driftlock/providers/anthropic_provider.py) —
  worked examples.

Add pricing for the new models in
[`driftlock/pricing.py`](driftlock/pricing.py) and tests mirroring
`tests/test_providers.py`.

## Adding a new policy rule

Subclass `BaseRule` and return a `RuleDecision`. Every built-in rule in
[`driftlock/policy.py`](driftlock/policy.py) is a worked example; see
[docs/policy-engine.md](docs/policy-engine.md#writing-a-custom-rule) for a
minimal template. Export it from `driftlock/__init__.py` and add tests
alongside `tests/test_policy.py`.

## PR guidelines

- **One feature or fix per PR.** Small, reviewable diffs merge faster.
- **Tests required.** New behavior needs new tests; bug fixes need a regression
  test.
- **`ruff` clean and `pytest` green** before you open the PR.
- **No new hard dependencies** in the core — use an optional extra.
- Update the relevant doc in [`docs/`](docs/) and `CHANGELOG.md` if you change
  public behavior.

By contributing you agree your work is licensed under the project's MIT license.
