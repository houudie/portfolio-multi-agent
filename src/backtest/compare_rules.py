"""Compare 4 règles d'allocation x 3 profils en validation (Hatim)."""
import pandas as pd

from src.agents.profile import PROFILES, apply_constraints
from src.agents.risk import RegimeDetector, min_variance_lw
from src.backtest.baselines import equal_weight, max_sharpe
from src.backtest.engine import compare, run_backtest
from src.backtest.forecast_strategies import top_k
from src.config import ROOT, load_config

DEFENSIVE = ["TLT", "IEF", "GLD"]


def constrained(rule, profile):
    c = PROFILES[profile]
    return lambda hist: apply_constraints(rule(hist), DEFENSIVE, c["min_defensif"], c["poids_max"])


def main():
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed"]
    prices = pd.read_csv(proc / "prices_clean.csv", index_col=0, parse_dates=True)
    preds = pd.read_csv(proc / "predictions" / "lightgbm_val.csv", index_col=0, parse_dates=True)
    universe = cfg["universe"]["stocks"] + cfg["universe"]["etfs"]
    px, spy = prices[universe], prices[cfg["universe"]["benchmark"]]
    start = pd.Timestamp(cfg["split"]["train_end"]) + pd.Timedelta(days=1)
    end = pd.Timestamp(cfg["split"]["val_end"])
    detector = RegimeDetector().fit(spy.loc[:cfg["split"]["train_end"]])

    def current_rule(hist):
        if detector.predict(spy.loc[:hist.index[-1]]) == "tendu":
            return min_variance_lw()(hist)
        return top_k(preds, 8)(hist)

    rules = {"A 1/N": equal_weight, "B Top 8 LightGBM": top_k(preds, 8),
             "C Markowitz": max_sharpe(), "D Règle actuelle": current_rule}
    results = [run_backtest(px, constrained(fn, p), start, end, name=f"{name} | {p}",
                            cost=cfg["backtest"]["transaction_cost"])
               for name, fn in rules.items() for p in PROFILES]
    table = compare(results)
    table.to_csv(ROOT / "reports" / "allocation_rules_validation.csv")
    print(table.to_string())
    print("\nSharpe moyen sur les 3 profils :")
    print(table["sharpe"].groupby(lambda s: s.split(" | ")[0]).mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()