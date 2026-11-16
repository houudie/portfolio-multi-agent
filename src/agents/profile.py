"""Agent Profil (membre D) : transforme un questionnaire en contraintes d'investissement.

Chaque réponse rapporte des points ; le total (0 à 20) donne un profil :
prudent (0-7), équilibré (8-14), dynamique (15-20).
"""
import pandas as pd

from src.agents.base import Agent, SharedState

QUESTIONS = {
    "horizon": {
        "texte": "Dans combien de temps aurez-vous besoin de cet argent ?",
        "choix": {"Moins de 2 ans": 0, "2 à 5 ans": 2, "5 à 10 ans": 3, "Plus de 10 ans": 4},
    },
    "reaction_baisse": {
        "texte": "Votre portefeuille perd 20 % en un mois. Que faites-vous ?",
        "choix": {"Je vends tout": 0, "Je vends une partie": 1, "J'attends": 3, "J'achète plus": 4},
    },
    "objectif": {
        "texte": "Quel est votre objectif principal ?",
        "choix": {"Protéger mon capital": 0, "Un revenu régulier": 1,
                  "Une croissance modérée": 3, "Une croissance maximale": 4},
    },
    "experience": {
        "texte": "Quelle est votre expérience des marchés financiers ?",
        "choix": {"Aucune": 0, "Livret ou assurance-vie": 1, "J'ai déjà acheté des actions": 3,
                  "J'investis régulièrement": 4},
    },
    "part_patrimoine": {
        "texte": "Quelle part de votre épargne représente ce placement ?",
        "choix": {"Plus de 75 %": 0, "50 à 75 %": 1, "25 à 50 %": 3, "Moins de 25 %": 4},
    },
}

PROFILES = {
    # part minimale en actifs défensifs (obligations + or) et poids maximal par actif
    "prudent":   {"min_defensif": 0.50, "poids_max": 0.15},
    "équilibré": {"min_defensif": 0.25, "poids_max": 0.20},
    "dynamique": {"min_defensif": 0.00, "poids_max": 0.30},
}


def score(answers: dict[str, str]) -> int:
    missing = set(QUESTIONS) - set(answers)
    if missing:
        raise ValueError(f"Réponses manquantes : {sorted(missing)}")
    return sum(QUESTIONS[q]["choix"][a] for q, a in answers.items())


def profile_from_score(s: int) -> str:
    return "prudent" if s <= 7 else "équilibré" if s <= 14 else "dynamique"


def apply_constraints(weights: pd.Series, defensive: list[str], min_defensive: float,
                      max_weight: float) -> pd.Series:
    """Impose la part défensive minimale puis le plafond par actif."""
    w = weights.clip(lower=0)
    w = w / w.sum()
    dfn = [a for a in defensive if a in w.index]
    risky = [a for a in w.index if a not in dfn]

    if dfn and w[dfn].sum() < min_defensive:
        base_dfn = w[dfn] if w[dfn].sum() > 0 else pd.Series(1.0, index=dfn)
        w[dfn] = min_defensive * base_dfn / base_dfn.sum()
        if w[risky].sum() > 0:
            w[risky] = (1 - min_defensive) * w[risky] / w[risky].sum()

    # plafond : on redistribue l'excédent sur les actifs encore sous le plafond
    max_weight = max(max_weight, 1.0 / len(w))
    for _ in range(100):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        room = (~over) & (w < max_weight)
        w[room] += excess * (w[room] + 1e-9) / (w[room] + 1e-9).sum()
    return w / w.sum()


class ProfileAgent(Agent):
    name = "Agent Profil"

    def run(self, state: SharedState) -> SharedState:
        s = score(state.profile["reponses"])
        name = profile_from_score(s)
        state.profile.update({"score": s, "profil": name, **PROFILES[name]})
        state.log.append(f"{self.name} : score {s}/20 -> profil {name}")
        return state
