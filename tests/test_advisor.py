"""Tests de l'agent Conseiller (Hatim)."""
import numpy as np
import pandas as pd

from src.agents.base import SharedState
from src.agents.pipeline import AdvisorAgent


def _state(forecast=True, regime="tendu"):
    weights = {"AAPL": 0.35, "MSFT": 0.15, "TLT": 0.3, "GLD": 0.2}
    f = pd.Series({"AAPL": 0.03, "MSFT": 0.01, "TLT": -0.01, "GLD": 0.0}) if forecast else None
    return SharedState(
        profile={"profil": "prudent", "score": 5, "min_defensif": 0.5, "poids_max": 0.35},
        weights=weights, forecasts={"rendement_prevu_21j": f},
        risk={"regime": regime, "cvar_%": -1.234}, data={"methode": "top 8"})


def test_explanation_mentions_profile_defensive_share_and_risk():
    text = AdvisorAgent().run(_state()).explanation
    assert "prudent" in text and "Part défensive" in text and "50 %" in text
    assert "1,2 %" in text            # CVaR au format français
    assert "agités" in text           # avertissement en régime tendu


def test_forecasts_are_shown_as_simple_returns_and_optional():
    text = AdvisorAgent().run(_state()).explanation
    assert f"AAPL ({100 * np.expm1(0.03):.1f}".replace(".", ",") in text
    text_without = AdvisorAgent().run(_state(forecast=False, regime="calme")).explanation
    assert "parts égales" in text_without and "calmes" in text_without