"""Customer churn analysis toolkit.

A package of importable, unit-testable modules. The Streamlit ``app.py`` is a
thin orchestration layer on top of these. Determinism is enforced through a
single global ``RANDOM_STATE`` threaded everywhere.
"""

RANDOM_STATE = 42

__all__ = ["RANDOM_STATE"]
