"""Pipeline multi-agent de bout en bout (membre D, version 1).

Ordre : Profil -> Données -> Prévision -> Risque -> Allocation -> Conseiller.
Chaque agent lit et enrichit le même `SharedState`.

Cette première version branche les briques déjà codées. D l'améliore ensuite
(meilleurs textes du Conseiller, choix de la stratégie, application Streamlit).

Démo : python -m src.agents.pipeline
"""
import pandas as pd

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
    """Règle v1 : régime tendu -> variance minimale (Ledoit-Wolf) ;
    sinon top 8 des prévisions, ou 1/N sans prévision. Puis contraintes du profil."""
    name = "Agent Allocation"

    def __init__(self, predictions: pd.DataFrame | None, lookback: int = 252):
        self.predictions, self.lookback = predictions, lookback

    def run(self, state):
        hist = state.data["prices"][state.data["universe"]].iloc[-self.lookback:]
        regime = state.risk["regime"]
        if regime == "tendu":
            method, w = "variance minimale (Ledoit-Wolf)", min_variance_lw()(hist)
        elif state.forecasts.get("rendement_prevu_21j") is not None:
            method, w = "top 8 des rendements prévus", top_k(self.predictions)(hist)
        else:
            method, w = "équipondéré (1/N)", equal_weight(hist)
        p = state.profile
        w = apply_constraints(w, DEFENSIVE, p["min_defensif"], p["poids_max"])
        state.weights = w.to_dict()
        state.risk.update(var_cvar(w, hist))
        state.data["methode"] = method
        state.log.append(f"{self.name} : méthode {method}")
        return state


class AdvisorAgent(Agent):
    name = "Agent Conseiller"

    def run(self, state):
        w = pd.Series(state.weights).sort_values(ascending=False)
        p, r = state.profile, state.risk
        top = ", ".join(f"{a} {100 * v:.0f} %" for a, v in w.head(5).items())
        defensive = 100 * w.reindex(DEFENSIVE).fillna(0).sum()
        state.explanation = (
            f"Votre profil est **{p['profil']}** (score {p['score']}/20) : au moins "
            f"{100 * p['min_defensif']:.0f} % en actifs défensifs et {100 * p['poids_max']:.0f} % maximum par actif.\n\n"
            f"Le marché est actuellement en régime **{r['regime']}**, d'où la méthode : {state.data['methode']}.\n\n"
            f"Principales positions : {top}. Part défensive (obligations et or) : {defensive:.0f} %.\n\n"
            f"Risque estimé : lors des 5 % pires jours de l'année passée, ce portefeuille aurait perdu "
            f"en moyenne {abs(r['cvar_%']):.1f} % par jour."
        )
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
