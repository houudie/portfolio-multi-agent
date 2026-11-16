"""Agent Prévision — modèles (étape 4, membre B).

On prédit `target` = rendement log des 21 prochains jours de bourse de chaque actif,
à partir des features causales de l'étape 2.

Modèles comparés, du plus simple au plus avancé :
- moyenne_historique : prédit toujours la moyenne du train (baseline)
- ridge              : régression linéaire régularisée (features standardisées)
- foret_aleatoire    : forêt d'arbres (scikit-learn)
- lightgbm, xgboost  : gradient boosting d'arbres

Anti-fuite : le StandardScaler du Ridge est dans un Pipeline, donc ajusté
sur le train seul ; la validation ne sert qu'à évaluer.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ROOT, load_config

TARGET = "target"
MODELS = ["moyenne_historique", "ridge", "foret_aleatoire", "lightgbm", "xgboost"]


def load_split(name: str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = ROOT / cfg["paths"]["processed"] / f"features_{name}.csv"
    return pd.read_csv(path, index_col=[0, 1], parse_dates=[0])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=TARGET), df[TARGET]


def make_model(name: str, seed: int = 42):
    if name == "moyenne_historique":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
    if name == "foret_aleatoire":
        return RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=200,
                                     max_features=0.5, n_jobs=-1, random_state=seed)
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=200, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=seed, verbose=-1)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                                min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                                random_state=seed)
    raise ValueError(f"Modèle inconnu : {name}")


def information_coefficient(y: pd.Series, pred: pd.Series) -> pd.Series:
    """Corrélation de rang (Spearman) entre prévision et réalité, date par date.

    C'est la mesure qui compte pour un portefeuille : sait-on CLASSER les actifs
    du meilleur au pire pour le mois suivant ?
    """
    df = pd.DataFrame({"y": y, "p": pred})

    def ic(g):
        if g["p"].nunique() < 2:
            return 0.0
        return g["y"].rank().corr(g["p"].rank())

    return df.groupby(level="date").apply(ic)


def evaluate(y: pd.Series, pred: pd.Series) -> dict:
    ic = information_coefficient(y, pred)
    err = y - pred
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
        "bon_sens_%": float(100 * (np.sign(y) == np.sign(pred)).mean()),
        "ic_moyen": float(ic.mean()),
        "ic_ecart_type": float(ic.std()),
        "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0,
        "ic_positif_%": float(100 * (ic > 0).mean()),
    }


def to_wide(pred: pd.Series) -> pd.DataFrame:
    """(date, actif) -> tableau date x actif, pratique pour le backtest."""
    return pred.unstack("ticker").sort_index()
