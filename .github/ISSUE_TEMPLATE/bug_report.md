---
name: Bug report
about: Report something that isn't working as expected
title: "[Bug] "
labels: bug
assignees: ''
---

## Describe the bug

A clear and concise description of what the bug is.

## Reproduction steps

Minimal steps (and code, if possible) to reproduce the behavior:

```python
# smallest snippet that reproduces the problem
```

1. ...
2. ...
3. ...

## Expected behavior

What you expected to happen instead.

## Environment

- Driftlock version: <!-- python -c "import driftlock; print(driftlock.__version__)" -->
- Python version: <!-- python --version -->
- OS:
- Provider / framework (OpenAI, Anthropic, LangChain, LangGraph):

## Additional context

Logs, tracebacks, or anything else that helps. Driftlock makes no network calls
in its test suite, so a self-contained repro is ideal.
