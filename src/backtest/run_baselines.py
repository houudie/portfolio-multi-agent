"""Lance les baselines sur la période d'entraînement et sur la validation (étape 3).

La période de TEST n'est pas utilisée ici : elle est réservée à l'étape 8.

Usage : python -m src.backtest.run_baselines
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.baselines import buy_and_hold, equal_weight, max_sharpe, min_variance
from src.backtest.engine import compare, run_backtest
from src.config import ROOT, load_config


def main() -> None:
    cfg = load_config()
    prices = pd.read_csv(ROOT / cfg["paths"]["processed"] / "prices_clean.csv",
                         index_col=0, parse_dates=True)
    bench = cfg["universe"]["benchmark"]
    universe = cfg["universe"]["stocks"] + cfg["universe"]["etfs"]
    cost = cfg["backtest"]["transaction_cost"]
    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_start = prices.index[252]  # 1 an d'historique pour les estimations
    periods = {
        "train": (train_start, pd.Timestamp(cfg["split"]["train_end"])),
        "validation": (pd.Timestamp(cfg["split"]["train_end"]) + pd.Timedelta(days=1),
                       pd.Timestamp(cfg["split"]["val_end"])),
    }
    strategies = {
        "1/N": (equal_weight, universe),
        f"Indice {bench}": (buy_and_hold(bench), [bench]),
        "Markowitz max Sharpe": (max_sharpe(), universe),
        "Variance minimale": (min_variance(), universe),
    }

    for period, (start, end) in periods.items():
        results = [run_backtest(prices[cols], fn, start, end, name=name, cost=cost)
                   for name, (fn, cols) in strategies.items()]
        table = compare(results)
        table.to_csv(ROOT / "reports" / f"baselines_{period}.csv")
        print(f"\n=== Baselines - {period} ({start.date()} -> {end.date()}) ===")
        print(table.to_string())

        fig, ax = plt.subplots(figsize=(11, 5))
        for res in results:
            (100 * res.wealth).plot(ax=ax, lw=1.4, label=res.name)
        ax.set(title=f"Valeur d'un portefeuille de 100 - baselines ({period})", xlabel="", ylabel="Valeur")
        ax.legend()
        fig.savefig(fig_dir / f"06_baselines_{period}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nTableaux : reports/baselines_*.csv - Figures : {fig_dir}/06_baselines_*.png")


if __name__ == "__main__":
    main()
