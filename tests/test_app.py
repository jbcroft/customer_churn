"""Headless Streamlit AppTest: the app boots and the full flow runs in-UI."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _boot():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    return at


def test_app_boots_without_exception():
    at = _boot()
    assert not at.exception


def test_demo_load_and_run_full_analysis():
    at = _boot()
    # click the "use bundled synthetic demo" button on the Upload stage
    demo = [b for b in at.button if "synthetic" in b.label.lower()]
    assert demo, "demo button missing"
    at.button(key=demo[0].key).click().run()
    assert not at.exception

    # navigate to the Target stage via the keyed radio so a target gets selected
    from churn.state import Stage

    at.radio(key="nav_radio").set_value(Stage.TARGET).run()
    assert not at.exception
    assert at.session_state["app"].target_col is not None

    # the "Run full analysis" sidebar button should now be enabled; click it
    run_btn = [b for b in at.sidebar.button if "Run full analysis" in b.label]
    assert run_btn
    at.button(key=run_btn[0].key).click().run()
    assert not at.exception
    # model + drivers should now exist in state
    state = at.session_state["app"]
    assert state.model_result is not None
    assert state.driver_table
