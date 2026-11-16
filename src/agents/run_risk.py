"""Évalue l'agent Risque en validation (membre A).

Usage : python -m src.agents.run_risk
Sorties : reports/risk_validation.csv, reports/figures/10_regimes.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.agents.risk import RegimeDetector, ledoit_wolf_cov, min_variance_lw, regime_switch
from src.backtest.baselines import equal_weight, max_sharpe, min_variance
from src.backtest.engine import compare, run_backtest
from src.config import ROOT, load_config

COLORS = {"calme": "#2a9d8f", "normal": "#e9c46a", "tendu": "#e76f51"}


def main() -> None:
    cfg = load_config()
    prices = pd.read_csv(ROOT / cfg["paths"]["processed"] / "prices_clean.csv",
                         index_col=0, parse_dates=True)
    bench = cfg["universe"]["benchmark"]
    universe = cfg["universe"]["stocks"] + cfg["universe"]["etfs"]
    px, spy = prices[universe], prices[bench]
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    start = train_end + pd.Timedelta(days=1)
    cost = cfg["backtest"]["transaction_cost"]

    # 1. Seuils de régime appris sur le train seul
    detector = RegimeDetector().fit(spy.loc[:train_end])
    print("Seuils de volatilité annualisée du SPY (appris sur le train) :")
    print(f"  calme  < {100 * detector.thresholds['calme_max']:.1f} %")
    print(f"  tendu >= {100 * detector.thresholds['tendu_min']:.1f} %")

    regimes = detector.series(spy.loc[:val_end])
    share = pd.DataFrame({
        "train_%": 100 * regimes.loc[:train_end].value_counts(normalize=True),
        "validation_%": 100 * regimes.loc[start:].value_counts(normalize=True),
    }).reindex(list(COLORS)).fillna(0).round(1)
    print("\nPart du temps passée dans chaque régime :")
    print(share.to_string())

    # 2. Ledoit-Wolf : intensité du rétrécissement à la fin du train
    _, shrink = ledoit_wolf_cov(px.loc[:train_end].iloc[-252:])
    print(f"\nRétrécissement Ledoit-Wolf fin 2019 : {shrink:.2f} (0 = covariance empirique, 1 = cible)")

    # 3. Figure : le SPY coloré par régime
    fig, ax = plt.subplots(figsize=(11, 4.5))
    s = spy.loc[regimes.index]
    for name, color in COLORS.items():
        ax.scatter(s.index[regimes == name], s[regimes == name], s=2, color=color, label=name)
    ax.axvline(train_end, color="black", ls="--", lw=1)
    ax.set(title="Régimes de marché détectés (volatilité du SPY sur 3 mois)", xlabel="", ylabel="Prix du SPY")
    ax.legend(markerscale=6)
    fig.savefig(ROOT / cfg["paths"]["figures"] / "10_regimes.png", dpi=150, bbox_inches="tight")

    # 4. Backtest en validation
    strategies = {
        "1/N": equal_weight,
        "Markowitz max Sharpe": max_sharpe(),
        "Variance minimale": min_variance(),
        "Variance minimale Ledoit-Wolf": min_variance_lw(),
        "Régime : 1/N, variance min si tendu": regime_switch(
            {"calme": equal_weight, "normal": equal_weight, "tendu": min_variance_lw()}, detector, spy),
        "Régime : 1/N, Markowitz si tendu": regime_switch(
            {"calme": equal_weight, "normal": equal_weight, "tendu": max_sharpe()}, detector, spy),
    }
    results = [run_backtest(px, fn, start, val_end, name=n, cost=cost) for n, fn in strategies.items()]
    table = compare(results).sort_values("sharpe", ascending=False)
    table.to_csv(ROOT / "reports" / "risk_validation.csv")
    print(f"\n=== Agent Risque - validation ({start.date()} -> {val_end.date()}) ===")
    print(table.to_string())


if __name__ == "__main__":
    main()
