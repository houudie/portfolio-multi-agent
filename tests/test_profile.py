"""Tests de l'agent Profil."""
import pandas as pd
import pytest

from src.agents.base import SharedState
from src.agents.profile import QUESTIONS, ProfileAgent, apply_constraints, profile_from_score, score


def _answers(level):
    # level 0 = première réponse partout, -1 = dernière
    return {q: list(d["choix"])[level] for q, d in QUESTIONS.items()}


def test_extreme_answers_give_extreme_profiles():
    assert profile_from_score(score(_answers(0))) == "prudent"
    assert profile_from_score(score(_answers(-1))) == "dynamique"


def test_missing_answer_is_refused():
    a = _answers(0)
    a.pop("horizon")
    with pytest.raises(ValueError):
        score(a)


def test_constraints_respected():
    w = pd.Series({"AAPL": 0.7, "MSFT": 0.3, "TLT": 0.0, "GLD": 0.0})
    out = apply_constraints(w, ["TLT", "GLD"], min_defensive=0.5, max_weight=0.3)
    assert abs(out.sum() - 1) < 1e-9
    assert out[["TLT", "GLD"]].sum() >= 0.5 - 1e-9
    assert (out <= 0.3 + 1e-9).all()


def test_agent_writes_profile():
    state = ProfileAgent().run(SharedState(profile={"reponses": _answers(-1)}))
    assert state.profile["profil"] == "dynamique" and state.profile["poids_max"] == 0.30
