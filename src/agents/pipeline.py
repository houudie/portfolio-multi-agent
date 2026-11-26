"""Pipeline multi-agent de bout en bout (membre D, version 1).

Ordre : Profil -> Données -> Prévision -> Risque -> Allocation -> Conseiller.
Chaque agent lit et enrichit le même `SharedState`.

Cette première version branche les briques déjà codées. D l'améliore ensuite
(meilleurs textes du Conseiller, choix de la stratégie, application Streamlit).

Démo : python -m src.agents.pipeline
"""
import pandas as pd
import numpy as np

from src.agents.base import Agent, Orchestrator, SharedState
from src.agents.profile import ProfileAgent, apply_constraints
from src.agents.risk import RegimeDetector, RiskAgent, min_variance_lw, var_cvar
from src.backtest.baselines import equal_weight
from src.backtest.forecast_strategies import latest_forecast, top_k
from src.config import ROOT, load_config

DEFENSIVE = ["TLT", "IEF", "GLD"]


class DataAgent(Agent):
    name = "Agent Données"

    def __init__(self, cfg: dict, as_of: str | None = None):
        self.cfg, self.as_of = cfg, as_of

    def run(self, state):
        prices = pd.read_csv(ROOT / self.cfg["paths"]["processed"] / "prices_clean.csv",
                             index_col=0, parse_dates=True)
        if self.as_of:
            prices = prices.loc[:self.as_of]          # on se place à une date passée
        state.data.update({"prices": prices, "date": prices.index[-1],
                           "universe": self.cfg["universe"]["stocks"] + self.cfg["universe"]["etfs"]})
        state.log.append(f"{self.name} : prix jusqu'au {prices.index[-1].date()}")
        return state


class ForecastAgent(Agent):
    name = "Agent Prévision"

    def __init__(self, predictions: pd.DataFrame | None):
        self.predictions = predictions

    def run(self, state):
        f = None if self.predictions is None else latest_forecast(self.predictions, state.data["date"])
        state.forecasts["rendement_prevu_21j"] = f
        state.log.append(f"{self.name} : " + ("prévisions disponibles" if f is not None
                                               else "pas de prévision à cette date -> repli sur 1/N"))
        return state


class AllocationAgent(Agent):
    """Règle retenue en validation (règle B) : poids égaux sur les 8 actifs au rendement
    prévu le plus élevé, ou 1/N sans prévision ; puis contraintes du profil.
    Le régime de marché est mesuré par l'agent Risque mais ne change plus les poids :
    passer en variance minimale en régime tendu dégradait le Sharpe (0,60 contre 1,12)."""
    name = "Agent Allocation"

    def __init__(self, predictions: pd.DataFrame | None, lookback: int = 252):
        self.predictions, self.lookback = predictions, lookback

    def run(self, state):
        hist = state.data["prices"][state.data["universe"]].iloc[-self.lookback:]
        if state.forecasts.get("rendement_prevu_21j") is not None:
            method, w = "top 8 des rendements prévus par LightGBM", top_k(self.predictions)(hist)
        else:
            method, w = "équipondéré (1/N), faute de prévision à cette date", equal_weight(hist)
        p = state.profile
        w = apply_constraints(w, DEFENSIVE, p["min_defensif"], p["poids_max"])
        state.weights = w.to_dict()
        state.risk.update(var_cvar(w, hist))
        state.data["methode"] = method
        state.log.append(f"{self.name} : méthode {method}")
        return state


class AdvisorAgent(Agent):
    """Rédige l'explication en français à partir de tout l'état partagé (Hatim)."""
    name = "Agent Conseiller"
    REGIMES = {
        "calme": "les marchés sont calmes",
        "normal": "la volatilité des marchés est dans la normale",
        "tendu": "les marchés sont agités : attendez-vous à de fortes variations à court terme",
    }

    @staticmethod
    def _pct(x: float, decimals: int = 1) -> str:
        return f"{x:.{decimals}f}".replace(".", ",") + " %"

    def run(self, state):
        pct = self._pct
        p, r = state.profile, state.risk
        w = pd.Series(state.weights)
        w = w[w > 1e-4].sort_values(ascending=False)
        defensive = 100 * w.reindex(DEFENSIVE).fillna(0).sum()

        if p["min_defensif"] > 0:
            contrainte = f"au moins {pct(100 * p['min_defensif'], 0)} en obligations et en or"
        else:
            contrainte = "aucune part minimale en obligations et en or"
        parts = [f"Votre profil est **{p['profil']}** (score {p['score']}/20) : {contrainte}, "
                 f"et {pct(100 * p['poids_max'], 0)} maximum par actif."]

        f = state.forecasts.get("rendement_prevu_21j")
        if f is not None:
            top = f.reindex(w.index).dropna().sort_values(ascending=False).head(3)
            prev = ", ".join(f"{a} ({pct(100 * np.expm1(v))})" for a, v in top.items())
            parts.append(f"Méthode : {state.data['methode']}. Le modèle retient les actifs dont il attend "
                         f"le meilleur rendement sur le mois à venir ; en tête : {prev}. Ces prévisions sont "
                         "incertaines : elles servent à classer les actifs, pas à garantir un gain.")
        else:
            parts.append(f"Méthode : {state.data['methode']}. Le capital est réparti à parts égales, "
                         "une stratégie simple et difficile à battre.")

        positions = ", ".join(f"{a} {pct(100 * v)}" for a, v in w.head(5).items())
        parts.append(f"Principales positions : {positions}. Part défensive (obligations et or) : {pct(defensive, 0)}.")
        regime = r.get("regime", "inconnu")
        parts.append(f"Contexte : {self.REGIMES.get(regime, regime)} (régime **{regime}**, "
                     "mesuré par la volatilité du S&P 500 sur 3 mois).")
        parts.append(f"Risque estimé : lors des 5 % pires jours de l'année passée, ce portefeuille aurait "
                     f"perdu en moyenne {pct(abs(r['cvar_%']))} par jour.")
        state.explanation = "\n\n".join(parts)
        state.log.append(f"{self.name} : explication rédigée")
        return state


def build_pipeline(cfg: dict, as_of: str | None = None, model: str = "lightgbm") -> Orchestrator:
    prices = pd.read_csv(ROOT / cfg["paths"]["processed"] / "prices_clean.csv", index_col=0, parse_dates=True)
    bench = cfg["universe"]["benchmark"]
    detector = RegimeDetector().fit(prices[bench].loc[:cfg["split"]["train_end"]])
    pred_dir = ROOT / cfg["paths"]["processed"] / "predictions"
    preds = None
    for name in [model, "xgboost", "foret_aleatoire", "ridge"]:   # premier fichier disponible
        if (pred_dir / f"{name}_val.csv").exists():
            preds = pd.read_csv(pred_dir / f"{name}_val.csv", index_col=0, parse_dates=True)
            break
    return Orchestrator([
        ProfileAgent(), DataAgent(cfg, as_of), ForecastAgent(preds),
        RiskAgent(detector, bench), AllocationAgent(preds), AdvisorAgent(),
    ])


def main() -> None:
    cfg = load_config()
    answers = {"horizon": "5 à 10 ans", "reaction_baisse": "J'attends",
               "objectif": "Une croissance modérée", "experience": "J'ai déjà acheté des actions",
               "part_patrimoine": "25 à 50 %"}
    state = build_pipeline(cfg, as_of="2021-06-30").run(SharedState(profile={"reponses": answers}))
    print("\n".join(state.log))
    print("\nPoids :")
    print((100 * pd.Series(state.weights)).sort_values(ascending=False).round(1).head(10).to_string())
    print("\n" + state.explanation.replace("**", ""))


if __name__ == "__main__":
    main()
