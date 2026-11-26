"""Évaluation finale sur la période de TEST (étape 8, Houda).

À lancer UNE SEULE FOIS, quand tous les choix (modèles, hyperparamètres, règle
d'allocation) ont été figés sur la validation. Relancer ce script pour « améliorer »
un résultat de test serait une fuite d'information.

Ce que fait le script :
1. Réentraîne le modèle de prévision retenu sur train + validation (jamais sur le test),
   prédit le test et écrit data/processed/predictions/<modele>_test.csv
   (utilisable ensuite par le pipeline et l'application sur des dates de test).
2. Mesure la qualité des prévisions sur le test (IC, RMSE...).
3. Backteste sur le test, avec le même moteur : baselines, agent Risque, stratégies
   avec prévisions, PPO (si un modèle entraîné existe) et la règle complète du
   pipeline pour les trois profils.

Usage :
    python -m src.evaluation.run_test                    # modèle de prévision : lightgbm
    python -m src.evaluation.run_test --model xgboost

Sorties :
    reports/test_forecast.csv, reports/test_results.csv, reports/test_profils.csv
    reports/figures/11_test_strategies.png, reports/figures/12_test_profils.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.agents.profile import PROFILES, apply_constraints
from src.agents.risk import RegimeDetector, min_variance_lw, regime_switch
from src.backtest.baselines import buy_and_hold, equal_weight, max_sharpe, min_variance
from src.backtest.engine import compare, run_backtest
from src.backtest.forecast_strategies import forecast_max_sharpe, latest_forecast, top_k
from src.config import ROOT, load_config
from src.models.forecast import evaluate, load_split, make_model, split_xy, to_wide

DEFENSIVE = ["TLT", "IEF", "GLD"]


# ------------------------------------------------------------------ prévisions
def forecast_test(cfg: dict, model_name: str) -> pd.DataFrame | None:
    """Réentraîne sur train + validation, prédit le test. Renvoie un tableau date x actif."""
    try:
        model = make_model(model_name, cfg["seed"])
    except ImportError as e:
        print(f"[ignoré] {model_name} : bibliothèque manquante ({e.name})")
        return None
    X_tr, y_tr = split_xy(pd.concat([load_split("train", cfg), load_split("val", cfg)]))
    X_te, y_te = split_xy(load_split("test", cfg))
    print(f"Prévision : {model_name} réentraîné sur train + validation ({len(X_tr)} lignes) ...", flush=True)
    model.fit(X_tr, y_tr)
    pred = pd.Series(model.predict(X_te), index=X_te.index, name=model_name)

    scores = pd.DataFrame([{"modele": model_name, **evaluate(y_te, pred)}]).set_index("modele").round(4)
    scores.to_csv(ROOT / "reports" / "test_forecast.csv")
    print("\n=== Qualité des prévisions - test ===")
    print(scores.to_string())

    wide = to_wide(pred)
    out = ROOT / cfg["paths"]["processed"] / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out / f"{model_name}_test.csv")
    return wide


# ------------------------------------------------------------------ règle du pipeline
def profile_strategy(profile: str, detector: RegimeDetector, spy: pd.Series,
                     preds: pd.DataFrame | None, k: int = 8):
    """Même règle que AllocationAgent (src/agents/pipeline.py), sous forme de stratégie.

    Régime tendu -> variance minimale Ledoit-Wolf ; sinon top k des prévisions
    (ou 1/N sans prévision) ; puis contraintes du profil.
    À garder synchronisée avec AllocationAgent si la règle change.
    """
    c = PROFILES[profile]

    def strategy(hist: pd.DataFrame) -> pd.Series:
        t = hist.index[-1]
        regime = detector.predict(spy.loc[:t])
        if regime == "tendu":
            w = min_variance_lw()(hist)
        elif preds is not None and latest_forecast(preds, t) is not None:
            w = top_k(preds, k)(hist)
        else:
            w = equal_weight(hist)
        return apply_constraints(w, DEFENSIVE, c["min_defensif"], c["poids_max"])
    return strategy


def ppo_strategy():
    path = ROOT / "models_saved" / "ppo_portfolio.zip"
    if not path.exists():
        print("[ignoré] PPO : models_saved/ppo_portfolio.zip absent (lancer src.models.rl.train_ppo)")
        return None
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ignoré] PPO : stable-baselines3 non installé")
        return None
    from src.models.rl.env import rl_strategy
    return rl_strategy(PPO.load(path))


# ------------------------------------------------------------------ figures
def plot(results, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for res in results:
        (100 * res.wealth).plot(ax=ax, lw=1.3, label=res.name)
    ax.set(title=title, xlabel="", ylabel="Valeur")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lightgbm", help="modèle de prévision retenu en validation")
    args = parser.parse_args()

    cfg = load_config()
    prices = pd.read_csv(ROOT / cfg["paths"]["processed"] / "prices_clean.csv",
                         index_col=0, parse_dates=True)
    bench = cfg["universe"]["benchmark"]
    universe = cfg["universe"]["stocks"] + cfg["universe"]["etfs"]
    px, spy = prices[universe], prices[bench]
    cost = cfg["backtest"]["transaction_cost"]
    start = pd.Timestamp(cfg["split"]["val_end"]) + pd.Timedelta(days=1)
    end = pd.Timestamp(cfg["dates"]["end"])
    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Seuils de régime : appris sur le train seul, comme en validation
    detector = RegimeDetector().fit(spy.loc[:cfg["split"]["train_end"]])
    preds = forecast_test(cfg, args.model)

    # 1. Toutes les stratégies, profil sans contrainte
    strategies = {
        "1/N": equal_weight,
        "Markowitz max Sharpe": max_sharpe(),
        "Variance minimale": min_variance(),
        "Variance minimale Ledoit-Wolf": min_variance_lw(),
        "Régime : 1/N, variance min si tendu": regime_switch(
            {"calme": equal_weight, "normal": equal_weight, "tendu": min_variance_lw()}, detector, spy),
    }
    if preds is not None:
        strategies[f"Top 8 - {args.model}"] = top_k(preds, k=8)
        strategies[f"Markowitz prévu - {args.model}"] = forecast_max_sharpe(preds)
    ppo = ppo_strategy()
    if ppo is not None:
        strategies["PPO"] = ppo

    results = [run_backtest(prices[[bench]], buy_and_hold(bench), start, end, name=f"Indice {bench}", cost=cost)]
    for name, fn in strategies.items():
        print(f"Backtest test : {name} ...", flush=True)
        results.append(run_backtest(px, fn, start, end, name=name, cost=cost))
    table = compare(results).sort_values("sharpe", ascending=False)
    table.to_csv(ROOT / "reports" / "test_results.csv")
    print(f"\n=== Toutes les stratégies - test ({start.date()} -> {end.date()}) ===")
    print(table.to_string())
    plot(results, "Valeur d'un portefeuille de 100 - test", fig_dir / "11_test_strategies.png")

    # 2. Le pipeline complet, pour chaque profil
    by_profile = [run_backtest(px, profile_strategy(p, detector, spy, preds), start, end,
                               name=f"Assistant - {p}", cost=cost) for p in PROFILES]
    by_profile.append(next(r for r in results if r.name == "1/N"))
    by_profile.append(results[0])  # indice
    table_p = compare(by_profile)
    table_p.to_csv(ROOT / "reports" / "test_profils.csv")
    print("\n=== Assistant multi-agent par profil - test ===")
    print(table_p.to_string())
    plot(by_profile, "Assistant multi-agent par profil - test", fig_dir / "12_test_profils.png")


if __name__ == "__main__":
    main()
