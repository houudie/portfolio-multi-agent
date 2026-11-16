"""Interface commune à tous les agents (détaillée à l'étape 5).

Chaque agent lit l'état partagé, fait son travail, et écrit son résultat
dans ce même état. L'orchestrateur appelle les agents dans l'ordre.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SharedState:
    """État commun que les agents se passent."""
    profile: dict[str, Any] = field(default_factory=dict)      # Agent Profil
    data: dict[str, Any] = field(default_factory=dict)         # Agent Données
    forecasts: dict[str, Any] = field(default_factory=dict)    # Agent Prévision
    risk: dict[str, Any] = field(default_factory=dict)         # Agent Risque
    weights: dict[str, float] = field(default_factory=dict)    # Agent Allocation
    explanation: str = ""                                      # Agent Conseiller
    log: list[str] = field(default_factory=list)


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def run(self, state: SharedState) -> SharedState:
        """Lit l'état, ajoute son résultat, renvoie l'état."""


class Orchestrator:
    def __init__(self, agents: list[Agent]):
        self.agents = agents

    def run(self, state: SharedState) -> SharedState:
        for agent in self.agents:
            state = agent.run(state)
        return state
