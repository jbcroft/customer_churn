"""Typed session-state container and pipeline-stage enum.

This module is deliberately free of any Streamlit import so the whole package
stays unit-testable headless. ``app.py`` stores a single :class:`AppState`
instance in ``st.session_state`` and calls the ``invalidate_*`` helpers when an
upstream choice changes, which clears downstream artifacts and forces a re-run.

The data-flow spine (single source of truth):

    UPLOAD -> PROFILE -> SELECT TARGET -> CLEAN -> TRANSFORM
           -> MODEL -> DRIVERS -> VISUALIZE -> REPORT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Stage(IntEnum):
    """Ordered pipeline stages. ``IntEnum`` so we can compare/advance with ``<``."""

    UPLOAD = 0
    PROFILE = 1
    TARGET = 2
    CLEAN = 3
    TRANSFORM = 4
    MODEL = 5
    DRIVERS = 6
    VISUALIZE = 7
    REPORT = 8

    @property
    def label(self) -> str:
        return {
            Stage.UPLOAD: "Upload",
            Stage.PROFILE: "Profile",
            Stage.TARGET: "Select target",
            Stage.CLEAN: "Clean",
            Stage.TRANSFORM: "Transform",
            Stage.MODEL: "Model",
            Stage.DRIVERS: "Drivers",
            Stage.VISUALIZE: "Visualize",
            Stage.REPORT: "Report",
        }[self]


@dataclass
class TransformationLog:
    """Human-readable audit trail of everything done to the user's data.

    Doubles as an audit panel in the UI and as an input to the AI report.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, stage: str, action: str, detail: str = "", **meta: Any) -> None:
        self.entries.append(
            {"stage": stage, "action": action, "detail": detail, "meta": meta}
        )

    def extend(self, other: "TransformationLog") -> None:
        self.entries.extend(other.entries)

    def as_lines(self) -> list[str]:
        lines = []
        for e in self.entries:
            line = f"[{e['stage']}] {e['action']}"
            if e["detail"]:
                line += f" — {e['detail']}"
            lines.append(line)
        return lines

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.entries)


@dataclass
class AppState:
    """Versioned artifacts for the linear, gated pipeline.

    Each stage writes its artifact here. Changing an upstream choice invalidates
    downstream artifacts (see the ``invalidate_*`` methods).
    """

    # --- UPLOAD / IO ---
    dataset_name: str | None = None
    raw_df: Any = None  # pandas.DataFrame
    type_overrides: dict[str, str] = field(default_factory=dict)  # col -> role

    # --- PROFILE ---
    profile: Any = None  # profiling.ProfileResult

    # --- TARGET ---
    target_col: str | None = None
    positive_class: Any = None  # value mapped to 1
    base_rate: float | None = None  # positive prevalence

    # --- CLEAN ---
    clean_df: Any = None
    cleaning_config: dict[str, Any] = field(default_factory=dict)

    # --- TRANSFORM ---
    feature_spec: Any = None  # features.FeatureSpec
    vif_table: Any = None

    # --- MODEL ---
    model_result: Any = None  # modeling.ModelResult
    threshold: float = 0.5

    # --- DRIVERS ---
    driver_table: Any = None  # pandas.DataFrame

    # --- VIZ ---
    figures: dict[str, Any] = field(default_factory=dict)

    # --- REPORT ---
    findings_payload: dict[str, Any] | None = None
    report_markdown: str | None = None

    # --- cross-cutting ---
    log: TransformationLog = field(default_factory=TransformationLog)

    # ------------------------------------------------------------------
    # Invalidation: clearing an upstream artifact wipes everything after it.
    # ------------------------------------------------------------------
    def invalidate_from(self, stage: Stage) -> None:
        """Clear all artifacts at ``stage`` and beyond."""
        if stage <= Stage.TARGET:
            self.target_col = None
            self.positive_class = None
            self.base_rate = None
        if stage <= Stage.CLEAN:
            self.clean_df = None
        if stage <= Stage.TRANSFORM:
            self.feature_spec = None
            self.vif_table = None
        if stage <= Stage.MODEL:
            self.model_result = None
        if stage <= Stage.DRIVERS:
            self.driver_table = None
        if stage <= Stage.VISUALIZE:
            self.figures = {}
        if stage <= Stage.REPORT:
            self.findings_payload = None
            self.report_markdown = None

    def furthest_ready_stage(self) -> Stage:
        """Highest stage for which a valid artifact exists."""
        if self.report_markdown is not None:
            return Stage.REPORT
        if self.figures:
            return Stage.VISUALIZE
        if self.driver_table is not None:
            return Stage.DRIVERS
        if self.model_result is not None:
            return Stage.MODEL
        if self.feature_spec is not None:
            return Stage.TRANSFORM
        if self.clean_df is not None:
            return Stage.CLEAN
        if self.target_col is not None:
            return Stage.TARGET
        if self.profile is not None:
            return Stage.PROFILE
        if self.raw_df is not None:
            return Stage.UPLOAD
        return Stage.UPLOAD
