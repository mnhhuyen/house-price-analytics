"""Smoke tests for the deployed valuation stack: the model artifact, the
shared serving layer, and the FastAPI service. A broken artifact can never
ship silently — these run on every change.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modeling.valuation_service import ValuationService

TYPICAL_HOME = {
    "Neighborhood": "NAmes", "GrLivArea": 1500, "TotalBsmtSF": 1000,
    "OverallQual": 5, "property_age": 30, "total_baths": 2.0, "GarageCars": 2,
}
PREMIUM_HOME = {
    "Neighborhood": "NridgHt", "GrLivArea": 3200, "TotalBsmtSF": 1600,
    "OverallQual": 9, "property_age": 3, "total_baths": 3.5, "GarageCars": 3,
    "KitchenQual": "Ex", "dist_school_km": 0.5,
}


@pytest.fixture(scope="module")
def service() -> ValuationService:
    return ValuationService()


def test_typical_home_estimate_is_plausible(service):
    r = service.predict(TYPICAL_HOME)
    assert 80_000 < r["estimate_usd"] < 300_000
    assert r["range_low_usd"] < r["estimate_usd"] < r["range_high_usd"]


def test_premium_home_worth_more_with_wider_dollar_interval(service):
    typical, premium = service.predict(TYPICAL_HOME), service.predict(PREMIUM_HOME)
    assert premium["estimate_usd"] > typical["estimate_usd"] * 1.5
    typical_width = typical["range_high_usd"] - typical["range_low_usd"]
    premium_width = premium["range_high_usd"] - premium["range_low_usd"]
    assert premium_width > typical_width  # honest uncertainty scales with value


def test_empty_input_falls_back_to_typical_home_defaults(service):
    r = service.predict({})
    assert 100_000 < r["estimate_usd"] < 250_000


def test_valuation_tool_renders_without_exception():
    # Executes the valuation page (catches ordering bugs like
    # set_page_config-not-first that a plain HTTP check misses).
    # Multipage entry point (streamlit_app.py) is a thin navigator; the
    # pages themselves are what users see.
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent
                               / "app" / "valuation_tool.py"),
                           default_timeout=30).run()
    assert not at.exception, f"page raised: {at.exception}"
    assert at.title[0].value.startswith("🏠")


def test_monitoring_dashboard_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent
                               / "app" / "monitoring_dashboard.py"),
                           default_timeout=30).run()
    assert not at.exception, f"page raised: {at.exception}"
    assert at.title[0].value.startswith("📈")
    # the verdict must come from the committed artifact, not a hardcoded claim
    verdicts = [m.value for m in at.metric]
    assert any("retrain since 2010-03" in str(v) for v in verdicts)


def test_multipage_entry_point_has_both_pages():
    """The deploy entry point must expose both demos in one process."""
    src = (Path(__file__).resolve().parent.parent
           / "app" / "streamlit_app.py").read_text()
    assert "valuation_tool.py" in src
    assert "monitoring_dashboard.py" in src
    assert "st.navigation" in src


def test_api_predict_and_health():
    from fastapi.testclient import TestClient
    from app.api.main import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    resp = client.post("/predict", json=TYPICAL_HOME)
    assert resp.status_code == 200
    body = resp.json()
    assert body["range_low_usd"] < body["estimate_usd"] < body["range_high_usd"]
    assert body["coverage"] == 0.8
    assert "dist_school_km" in body["defaults_applied"]

    # unknown category -> clear 422, not a silent wrong answer
    assert client.post("/predict", json={"Neighborhood": "Hanoi"}).status_code == 422
