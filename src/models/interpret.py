"""Interprétabilité de l'agent Prévision (étape 4, Rachid).

Trois analyses complémentaires, toutes sur la VALIDATION (le test reste intact) :
1. SHAP (TreeExplainer) : contribution de chaque variable à chaque prévision.
   - vue globale : importance moyenne |SHAP| + nuage (beeswarm)
   - dépendance : comment la prévision varie avec les 2 variables principales
   - vue locale : décomposition d'UNE prévision (un actif, une date)
2. Importance par permutation : perte d'IC quand on mélange une variable.
   Mesure ce dont le modèle dépend vraiment HORS échantillon.
3. Analyse des erreurs : IC mois par mois, mois les pires, IC par actif.

À lancer APRÈS python -m src.models.train_forecast (lit models_saved/<modele>.joblib).
Usage : python -m src.models.interpret                 # lightgbm
        python -m src.models.interpret --model xgboost

Sorties : reports/shap_importance.csv, reports/permutation_importance.csv,
          reports/ic_mensuel_validation.csv, reports/ic_par_actif_validation.csv,
          reports/figures/13_shap_importance.png ... 17_ic_mensuel.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.models.forecast import information_coefficient, load_split, split_xy

# Noms lisibles pour les figures et le rapport
LABELS = {
    "mom_21": "Momentum 1 mois", "mom_63": "Momentum 3 mois", "mom_126": "Momentum 6 mois",
    "vol_21": "Volatilité 1 mois", "vol_63": "Volatilité 3 mois", "rsi_14": "RSI 14 jours",
    "drawdown_252": "Drawdown 1 an", "mkt_mom_21": "Momentum SPY 1 mois",
    "mkt_vol_21": "Volatilité SPY 1 mois", "taux_10_ans": "Taux 10 ans", "vix": "VIX",
    "taux_directeur": "Taux directeur", "vix_chg_21": "Variation VIX 1 mois",
}


def label(col: str) -> str:
    return LABELS.get(col, col)


# ------------------------------------------------------------------ permutation
def permutation_ic(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5,
                   seed: int = 42) -> pd.DataFrame:
    """Baisse de l'IC moyen quand on mélange chaque variable (moyenne sur n_repeats).

    > 0 : le modèle s'appuie sur cette variable et elle l'aide hors échantillon.
    ~ 0 : variable inutile.  < 0 : variable qui nuit (le modèle a surappris dessus).
    """
    rng = np.random.default_rng(seed)
    base = information_coefficient(y, pd.Series(model.predict(X), index=X.index)).mean()
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            ic = information_coefficient(y, pd.Series(model.predict(Xp), index=X.index)).mean()
            drops.append(base - ic)
        rows.append({"variable": col, "baisse_ic": float(np.mean(drops)),
                     "ecart_type": float(np.std(drops))})
    out = pd.DataFrame(rows).set_index("variable").sort_values("baisse_ic", ascending=False)
    out.attrs["ic_reference"] = float(base)
    return out


# ------------------------------------------------------------------ erreurs
def monthly_ic(y: pd.Series, pred: pd.Series) -> pd.Series:
    ic = information_coefficient(y, pred)
    return ic.groupby(ic.index.to_period("M")).mean()


def ic_by_asset(y: pd.Series, pred: pd.Series) -> pd.Series:
    """Corrélation de rang dans le temps, actif par actif."""
    df = pd.DataFrame({"y": y, "p": pred})
    return df.groupby(level="ticker").apply(lambda g: g["y"].rank().corr(g["p"].rank())).sort_values()


# ------------------------------------------------------------------ SHAP
def shap_values(model, X: pd.DataFrame, max_rows: int, seed: int):
    import shap

    sample = X.sample(min(max_rows, len(X)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    return explainer, sample, explainer(sample)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--shap-rows", type=int, default=3000, help="lignes échantillonnées pour SHAP")
    args = parser.parse_args()

    cfg = load_config()
    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = ROOT / "models_saved" / f"{args.model}.joblib"
    if not path.exists():
        raise SystemExit(f"{path} absent : lancer d'abord python -m src.models.train_forecast")
    model = joblib.load(path)
    X_va, y_va = split_xy(load_split("val", cfg))
    pred = pd.Series(model.predict(X_va), index=X_va.index)

    # 1. SHAP -------------------------------------------------------------
    import shap

    explainer, sample, sv = shap_values(model, X_va, args.shap_rows, cfg["seed"])
    sv.feature_names = [label(c) for c in sample.columns]
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=sample.columns, name="shap_moyen_abs")
    imp = imp.sort_values(ascending=False)
    imp.to_csv(ROOT / "reports" / "shap_importance.csv")

    plt.figure()
    shap.plots.bar(sv, max_display=14, show=False)
    plt.title(f"Importance SHAP moyenne - {args.model} (validation)")
    plt.savefig(fig_dir / "13_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.plots.beeswarm(sv, max_display=14, show=False)
    plt.title("Effet de chaque variable sur le rendement prévu à 21 jours")
    plt.savefig(fig_dir / "14_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, imp.index[:2]):
        j = list(sample.columns).index(col)
        ax.scatter(sample[col], sv.values[:, j], s=4, alpha=0.4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xlabel=label(col), ylabel="Valeur SHAP (effet sur la prévision)",
               title=f"Dépendance : {label(col)}")
    fig.tight_layout()
    fig.savefig(fig_dir / "15_shap_dependance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # vue locale : la prévision la plus élevée de la dernière date de l'échantillon
    last_date = sample.index.get_level_values("date").max()
    rows_last = np.where(sample.index.get_level_values("date") == last_date)[0]
    i = rows_last[np.argmax(model.predict(sample.iloc[rows_last]))]
    ticker = sample.index[i][1]
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    plt.title(f"Pourquoi {ticker} le {last_date.date()} ? (prévision la plus haute ce jour-là)")
    plt.savefig(fig_dir / "16_shap_local.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"\n=== SHAP - {args.model} : importance moyenne |SHAP| (validation, {len(sample)} lignes) ===")
    print(imp.rename(index=label).round(5).to_string())
    print(f"\nExplication locale : {ticker} le {last_date.date()} -> figure 16")

    # 2. Permutation ------------------------------------------------------
    print("\nImportance par permutation (IC en validation) ...", flush=True)
    perm = permutation_ic(model, X_va, y_va, seed=cfg["seed"])
    perm.to_csv(ROOT / "reports" / "permutation_importance.csv")
    print(f"IC de référence : {perm.attrs['ic_reference']:.4f}")
    print(perm.rename(index=label).round(4).to_string())

    # 3. Erreurs ----------------------------------------------------------
    m_ic = monthly_ic(y_va, pred)
    m_ic.to_csv(ROOT / "reports" / "ic_mensuel_validation.csv", header=["ic_moyen"])
    a_ic = ic_by_asset(y_va, pred)
    a_ic.to_csv(ROOT / "reports" / "ic_par_actif_validation.csv", header=["ic"])

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = np.where(m_ic.values >= 0, "#2a9d8f", "#e76f51")
    ax.bar(m_ic.index.astype(str), m_ic.values, color=colors)
    ax.axhline(m_ic.mean(), color="black", ls="--", lw=1, label=f"moyenne {m_ic.mean():.3f}")
    ax.set(title=f"IC mensuel en validation - {args.model}", ylabel="IC moyen du mois")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    fig.savefig(fig_dir / "17_ic_mensuel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Analyse des erreurs ===")
    print(f"Mois avec IC positif : {100 * (m_ic > 0).mean():.0f} % ({(m_ic > 0).sum()}/{len(m_ic)})")
    print("3 pires mois :", ", ".join(f"{p} ({v:.3f})" for p, v in m_ic.nsmallest(3).items()))
    print("3 meilleurs mois :", ", ".join(f"{p} ({v:.3f})" for p, v in m_ic.nlargest(3).items()))
    print("Actifs les moins bien classés :", ", ".join(f"{t} ({v:.2f})" for t, v in a_ic.head(3).items()))
    print("Actifs les mieux classés :", ", ".join(f"{t} ({v:.2f})" for t, v in a_ic.tail(3).items()))


if __name__ == "__main__":
    main()
