"""Entraîne les modèles de prévision sur le train et les évalue sur la validation.

Usage : python -m src.models.train_forecast
Sorties :
- reports/forecast_validation.csv           (tableau de comparaison)
- reports/figures/07_forecast_ic.png         (IC moyen par modèle)
- data/processed/predictions/<modele>_val.csv (prévisions date x actif)
- models_saved/<modele>.joblib               (modèles entraînés)
"""
import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ROOT, load_config
from src.models.forecast import MODELS, evaluate, load_split, make_model, split_xy, to_wide


def main() -> None:
    cfg = load_config()
    X_tr, y_tr = split_xy(load_split("train", cfg))
    X_va, y_va = split_xy(load_split("val", cfg))
    pred_dir = ROOT / cfg["paths"]["processed"] / "predictions"
    model_dir = ROOT / "models_saved"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    rows = []
    for name in MODELS:
        try:
            model = make_model(name, cfg["seed"])
        except ImportError as e:
            print(f"[ignoré] {name} : bibliothèque manquante ({e.name})")
            continue
        print(f"Entraînement : {name} ...", flush=True)
        model.fit(X_tr, y_tr)
        pred = pd.Series(model.predict(X_va), index=X_va.index, name=name)
        rows.append({"modele": name, **evaluate(y_va, pred)})
        to_wide(pred).to_csv(pred_dir / f"{name}_val.csv")
        joblib.dump(model, model_dir / f"{name}.joblib")

    table = pd.DataFrame(rows).set_index("modele").round(4)
    table.to_csv(ROOT / "reports" / "forecast_validation.csv")
    print("\n=== Prévision du rendement à 21 jours - validation ===")
    print(table.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    table["ic_moyen"].sort_values().plot.barh(ax=ax)
    ax.axvline(0, color="black", lw=1)
    ax.set(title="Information Coefficient moyen en validation (plus haut = meilleur classement)",
           xlabel="IC moyen (corrélation de rang)", ylabel="")
    fig.savefig(ROOT / cfg["paths"]["figures"] / "07_forecast_ic.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
