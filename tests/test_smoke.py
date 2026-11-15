"""Tests de base : la config est cohérente et l'orchestrateur fonctionne."""
import pandas as pd

from src.agents.base import Agent, Orchestrator, SharedState
from src.config import load_config


def test_config_split_is_chronological():
    cfg = load_config()
    start = pd.Timestamp(cfg["dates"]["start"])
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    end = pd.Timestamp(cfg["dates"]["end"])
    assert start < train_end < val_end < end


def test_orchestrator_runs_agents_in_order():
    class Dummy(Agent):
        def __init__(self, name):
            self.name = name

        def run(self, state):
            state.profile.setdefault("ordre", []).append(self.name)
            return state

    state = Orchestrator([Dummy("a"), Dummy("b")]).run(SharedState())
    assert state.profile["ordre"] == ["a", "b"]
