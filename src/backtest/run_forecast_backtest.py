"""Backtest en validation des stratégies basées sur les prévisions (étape 4).

À lancer APRÈS python -m src.models.train_forecast
Usage : python -m src.backtest.run_forecast_backtest
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.baselines import equal_weight, max_sharpe
from src.backtest.engine import compare, run_backtest
from src.backtest.forecast_strategies import forecast_max_sharpe, top_k
from src.config import ROOT, load_config


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed"]
    prices = pd.read_csv(proc / "prices_clean.csv", index_col=0, parse_dates=True)
    universe = cfg["universe"]["stocks"] + cfg["universe"]["etfs"]
    px = prices[universe]
    cost = cfg["backtest"]["transaction_cost"]
    start = pd.Timestamp(cfg["split"]["train_end"]) + pd.Timedelta(days=1)
    end = pd.Timestamp(cfg["split"]["val_end"])

    strategies = {"1/N": equal_weight, "Markowitz max Sharpe": max_sharpe()}
    for path in sorted((proc / "predictions").glob("*_val.csv")):
        model = path.stem.replace("_val", "")
        if model == "moyenne_historique":
            continue  # prévision constante : aucun classement possible
        preds = pd.read_csv(path, index_col=0, parse_dates=True)
        strategies[f"Top 8 - {model}"] = top_k(preds, k=8)
        strategies[f"Markowitz prévu - {model}"] = forecast_max_sharpe(preds)

    results = [run_backtest(px, fn, start, end, name=name, cost=cost) for name, fn in strategies.items()]
    table = compare(results).sort_values("sharpe", ascending=False)
    table.to_csv(ROOT / "reports" / "forecast_backtest_validation.csv")
    print(f"\n=== Stratégies avec prévisions - validation ({start.date()} -> {end.date()}) ===")
    print(table.to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    for res in results:
        (100 * res.wealth).plot(ax=ax, lw=1.2, label=res.name)
    ax.set(title="Valeur d'un portefeuille de 100 - stratégies avec prévisions (validation)",
           xlabel="", ylabel="Valeur")
    ax.legend(fontsize=8)
    fig.savefig(ROOT / cfg["paths"]["figures"] / "08_forecast_backtest_validation.png",
                dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
