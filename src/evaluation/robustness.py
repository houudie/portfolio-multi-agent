"""Analyses de robustesse (étape 8, Houda).

Ces analyses ne servent à CHOISIR aucun modèle : tous les choix ont été figés en
validation et le test a déjà été évalué une fois (run_test.py). Elles mesurent la
confiance qu'on peut accorder aux écarts observés.

1. Significativité : intervalle de confiance à 95 % du ratio de Sharpe de chaque
   stratégie, et de l'écart de Sharpe avec 1/N soumis aux MÊMES contraintes de profil,
   par bootstrap par blocs (blocs de 21 jours : on garde la dépendance temporelle).
2. Sensibilité aux frais : Sharpe pour des frais de 0 %, 0,1 %, 0,25 % et 0,5 %.
3. Comparaison juste sur le test : assistant contre 1/N contraint, profil par profil.
4. Stabilité des prévisions : IC de LightGBM année par année (validation et test).
5. Diagnostic a posteriori du changement de distribution : part des jours du test où
   les taux sortent de la plage vue à l'entraînement, et IC d'un LightGBM réentraîné
   SANS variables macro. Ce modèle n'est PAS retenu : il sert seulement à expliquer
   l'échec du modèle officiel (un choix fait après avoir vu le test serait une fuite).

Usage : python -m src.evaluation.robustness
Sorties : reports/robustesse_*.csv, reports/ic_par_annee.csv, reports/diagnostic_macro.csv,
          reports/figures/18_robustesse_sharpe.png, 19_robustesse_frais.png, 20_ic_par_annee.png,
          21_taux_hors_plage.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.profile import PROFILES, apply_constraints
from src.backtest.baselines import buy_and_hold, equal_weight, max_sharpe
from src.backtest.engine import BacktestResult, run_backtest
from src.config import ROOT, load_config
from src.evaluation.run_test import DEFENSIVE, profile_strategy
from src.models.forecast import information_coefficient, load_split, make_model, split_xy

MACRO = ["taux_10_ans", "vix", "taux_directeur", "vix_chg_21"]

PERIODS = {"validation": ("2020-01-01", "2021-12-31", "val"),
           "test": ("2022-01-01", "2025-12-31", "test")}
COSTS = [0.0, 0.001, 0.0025, 0.005]


# ------------------------------------------------------------------ statistiques
def sharpe(r: np.ndarray, periods: int = 252) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def block_bootstrap_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap par blocs circulaires : n_boot tirages de n indices, par blocs de `block` jours."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def sharpe_ci(returns: pd.DataFrame, reference: str | None = None, block: int = 21,
              n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Sharpe et IC à 95 % de chaque colonne ; écart avec `reference` (tirages APPARIÉS :
    les mêmes jours pour les deux stratégies, d'où un écart bien plus précis)."""
    rng = np.random.default_rng(seed)
    x = returns.dropna().values
    idx = block_bootstrap_indices(len(x), block, n_boot, rng)
    boot = np.stack([[sharpe(x[i, j]) for j in range(x.shape[1])] for i in idx])  # n_boot x k
    rows = []
    for j, col in enumerate(returns.columns):
        row = {"strategie": col, "sharpe": sharpe(x[:, j]),
               "sharpe_ic_bas": float(np.percentile(boot[:, j], 2.5)),
               "sharpe_ic_haut": float(np.percentile(boot[:, j], 97.5))}
        if reference is not None and col != reference:
            k = list(returns.columns).index(reference)
            diff = boot[:, j] - boot[:, k]
            row.update({"ecart_vs_ref": sharpe(x[:, j]) - sharpe(x[:, k]),
                        "ecart_ic_bas": float(np.percentile(diff, 2.5)),
                        "ecart_ic_haut": float(np.percentile(diff, 97.5)),
                        "p_valeur": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))})
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategie")


# ------------------------------------------------------------------ stratégies
def equal_weight_profile(profile: str):
    c = PROFILES[profile]
    return lambda hist: apply_constraints(equal_weight(hist), DEFENSIVE, c["min_defensif"], c["poids_max"])


def strategies(preds: pd.DataFrame) -> dict:
    out = {}
    for p in PROFILES:
        out[f"Assistant ({p})"] = profile_strategy(p, None, None, preds)
        out[f"1/N ({p})"] = equal_weight_profile(p)
    out["1/N"] = equal_weight
    out["Markowitz max Sharpe"] = max_sharpe()
    return out


def run_period(prices: pd.DataFrame, cfg: dict, start: str, end: str, preds: pd.DataFrame,
               cost: float) -> dict[str, BacktestResult]:
    universe = cfg["universe"]["stocks"] + cfg["universe"]["etfs"]
    bench = cfg["universe"]["benchmark"]
    res = {n: run_backtest(prices[universe], fn, start, end, name=n, cost=cost)
           for n, fn in strategies(preds).items()}
    res[f"Indice {bench}"] = run_backtest(prices[[bench]], buy_and_hold(bench), start, end,
                                          name=f"Indice {bench}", cost=cost)
    return res


# ------------------------------------------------------------------ diagnostic
def out_of_range_share(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Part des lignes du test dont la valeur sort de la plage [min, max] du train."""
    return pd.Series({c: float(((test[c] > train[c].max()) | (test[c] < train[c].min())).mean())
                      for c in cols})


def macro_diagnostic(cfg: dict) -> pd.DataFrame:
    tr = pd.concat([load_split("train", cfg), load_split("val", cfg)])
    te = load_split("test", cfg)
    X, y = split_xy(tr)
    Xt, yt = split_xy(te)
    rows = []
    for label, cols in [("Modèle officiel (toutes variables)", list(X.columns)),
                        ("Diagnostic : sans variables macro", [c for c in X.columns if c not in MACRO])]:
        model = make_model("lightgbm", cfg["seed"]).fit(X[cols], y)
        ic = information_coefficient(yt, pd.Series(model.predict(Xt[cols]), index=Xt.index))
        row = {"modele": label, "ic_test": ic.mean()}
        row.update({f"ic_{year}": v for year, v in ic.groupby(ic.index.year).mean().items()})
        rows.append(row)
    out = pd.DataFrame(rows).set_index("modele")
    share = out_of_range_share(tr, te, ["taux_10_ans", "taux_directeur", "vix"])
    out.attrs["hors_plage"] = share
    out.attrs["taux"] = pd.concat([tr, te])[["taux_10_ans", "taux_directeur"]].groupby(level="date").first()
    out.attrs["max_train"] = tr[["taux_10_ans", "taux_directeur"]].max()
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed"]
    fig_dir = ROOT / cfg["paths"]["figures"]
    prices = pd.read_csv(proc / "prices_clean.csv", index_col=0, parse_dates=True)
    preds = {k: pd.read_csv(proc / "predictions" / f"lightgbm_{s}.csv", index_col=0, parse_dates=True)
             for k, (_, _, s) in PERIODS.items()}

    # 1 + 3. Sharpe, intervalles de confiance, comparaison juste par profil -------------
    base_cost = cfg["backtest"]["transaction_cost"]
    boot_rows, all_results = [], {}
    for period, (start, end, _) in PERIODS.items():
        print(f"Backtests {period} ...", flush=True)
        res = run_period(prices, cfg, start, end, preds[period], base_cost)
        all_results[period] = res
        rets = pd.DataFrame({n: r.returns for n, r in res.items()})
        for p in PROFILES:   # assistant contre 1/N avec les mêmes contraintes
            t = sharpe_ci(rets[[f"Assistant ({p})", f"1/N ({p})"]], reference=f"1/N ({p})")
            boot_rows.append(t.assign(periode=period, comparaison=f"1/N ({p})"))
        t = sharpe_ci(rets[["Markowitz max Sharpe", f"Indice {cfg['universe']['benchmark']}", "1/N"]], reference="1/N")
        boot_rows.append(t.assign(periode=period, comparaison="1/N"))
    boot = pd.concat(boot_rows).reset_index()
    boot = boot.drop_duplicates(subset=["strategie", "periode"], keep="first").round(3)
    boot.to_csv(ROOT / "reports" / "robustesse_bootstrap.csv", index=False)
    print("\n=== Sharpe, IC à 95 % (bootstrap par blocs de 21 jours) et écart avec 1/N ===")
    print(boot.to_string(index=False))

    # 2. Sensibilité aux frais ------------------------------------------------------------
    fee_rows = []
    for period, (start, end, _) in PERIODS.items():
        for cost in COSTS:
            print(f"Frais {100 * cost:.2f} % - {period} ...", flush=True)
            res = all_results[period] if cost == base_cost else run_period(prices, cfg, start, end, preds[period], cost)
            for n, r in res.items():
                fee_rows.append({"periode": period, "frais_%": 100 * cost, "strategie": n,
                                 "sharpe": sharpe(r.returns.values)})
    fees = pd.DataFrame(fee_rows).pivot_table(index=["periode", "strategie"], columns="frais_%", values="sharpe").round(2)
    fees.to_csv(ROOT / "reports" / "robustesse_frais.csv")
    print("\n=== Sharpe selon les frais de transaction (%) ===")
    print(fees.to_string())

    # 4. IC année par année ---------------------------------------------------------------
    ic_rows = []
    for period, (_, _, split) in PERIODS.items():
        _, y = split_xy(load_split(split, cfg))
        p = preds[period].stack().rename("pred")
        p.index.names = ["date", "ticker"]
        df = pd.concat([y, p], axis=1, join="inner")
        ic = information_coefficient(df["target"], df["pred"])
        for year, v in ic.groupby(ic.index.year):
            ic_rows.append({"annee": year, "periode": period, "ic_moyen": v.mean(),
                            "ic_positif_%": 100 * (v > 0).mean(), "jours": len(v)})
    ic_year = pd.DataFrame(ic_rows).round(3)
    ic_year.to_csv(ROOT / "reports" / "ic_par_annee.csv", index=False)
    print("\n=== IC de LightGBM année par année ===")
    print(ic_year.to_string(index=False))

    # 5. Diagnostic macro -----------------------------------------------------------------
    print("\nDiagnostic macro (réentraînement sans variables macro) ...", flush=True)
    diag = macro_diagnostic(cfg)
    diag.round(3).to_csv(ROOT / "reports" / "diagnostic_macro.csv")
    print("Part des jours du test hors de la plage vue à l'entraînement :")
    print((100 * diag.attrs["hors_plage"]).round(0).to_string())
    print(diag.round(3).to_string())

    taux, mx = diag.attrs["taux"], diag.attrs["max_train"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for col, color, label in [("taux_10_ans", "#0F766E", "Taux à 10 ans"), ("taux_directeur", "#C9A227", "Taux directeur")]:
        ax.plot(taux.index, taux[col], color=color, lw=1.3, label=label)
        ax.axhline(mx[col], color=color, ls=":", lw=1)
    ax.axvspan(pd.Timestamp("2022-01-01"), taux.index.max(), color="#B4413C", alpha=0.07, label="Test")
    ax.set(title="Les taux du test dépassent tout ce que le modèle a vu à l'entraînement (pointillés : maximum train + validation)",
           ylabel="%")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(fig_dir / "21_taux_hors_plage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figures -----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    order = [f"Assistant ({p})" for p in PROFILES] + [f"1/N ({p})" for p in PROFILES] + ["Markowitz max Sharpe", "1/N"]
    for ax, period in zip(axes, PERIODS):
        b = boot[boot["periode"] == period].set_index("strategie").reindex(order).dropna(subset=["sharpe"])
        y = np.arange(len(b))[::-1]
        colors = ["#C9A227" if s.startswith("Assistant") else "#0F766E" for s in b.index]
        ax.errorbar(b["sharpe"], y, xerr=[b["sharpe"] - b["sharpe_ic_bas"], b["sharpe_ic_haut"] - b["sharpe"]],
                    fmt="none", ecolor="#7C8B8F", capsize=3)
        ax.scatter(b["sharpe"], y, c=colors, zorder=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set(yticks=y, yticklabels=b.index, title=f"{period.capitalize()} : Sharpe et IC à 95 %", xlabel="Ratio de Sharpe")
    fig.tight_layout()
    fig.savefig(fig_dir / "18_robustesse_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, period in zip(axes, PERIODS):
        f = fees.loc[period]
        for name in [f"Assistant ({p})" for p in PROFILES] + ["1/N", "Markowitz max Sharpe"]:
            ax.plot(f.columns, f.loc[name], marker="o", label=name,
                    color="#C9A227" if name.startswith("Assistant") else None,
                    alpha=0.5 + 0.5 * (name.endswith("dynamique)") or not name.startswith("Assistant")))
        ax.set(title=f"{period.capitalize()} : Sharpe selon les frais", xlabel="Frais par transaction (%)", ylabel="Sharpe")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "19_robustesse_frais.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = ["#0F766E" if v >= 0 else "#B4413C" for v in ic_year["ic_moyen"]]
    ax.bar(ic_year["annee"].astype(str), ic_year["ic_moyen"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(1.5, color="#7C8B8F", ls="--", lw=1)
    ax.text(1.55, ax.get_ylim()[1] * 0.85, "début du test", fontsize=9, color="#7C8B8F")
    ax.set(title="IC moyen de LightGBM par année", ylabel="IC moyen")
    fig.savefig(fig_dir / "20_ic_par_annee.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
