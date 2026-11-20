"""Préparation des données (étape 2).

Ordre des opérations, pensé pour éviter toute fuite de données :
1. Nettoyage des prix (dates communes, trous de cotation).
2. Découpage CHRONOLOGIQUE train / validation / test, défini par config.yaml.
3. Features causales : la valeur à la date t n'utilise que des données <= t.
4. Cible : rendement des `horizon` prochains jours (seule colonne tournée vers le futur).
5. Purge : on retire les `horizon` derniers jours du train et de la validation,
   sinon leur cible chevaucherait la période suivante.

Aucun scaler ni modèle n'est ajusté ici : ils le seront sur le train seul (étape 4).

Usage : python -m src.data.prepare
"""
import json

import numpy as np
import pandas as pd

from src.config import ROOT, load_config


# ---------------------------------------------------------------- chargement
def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = ROOT / cfg["paths"]["raw"]
    prices = pd.read_csv(raw / "prices.csv", index_col=0, parse_dates=True).sort_index()
    macro = pd.read_csv(raw / "macro.csv", index_col=0, parse_dates=True).sort_index()
    return prices, macro


# ---------------------------------------------------------------- nettoyage
def clean_prices(prices: pd.DataFrame, max_gap: int = 5) -> tuple[pd.DataFrame, dict]:
    """Garde la période où tous les actifs existent, comble les petits trous."""
    report = {
        "valeurs_manquantes_brutes": {k: int(v) for k, v in prices.isna().sum().items() if v > 0},
        "premiere_date_par_actif": {k: str(v.date()) for k, v in prices.apply(pd.Series.first_valid_index).items()},
    }
    prices = prices.dropna(axis=1, how="all")
    common_start = prices.apply(pd.Series.first_valid_index).max()
    prices = prices.loc[common_start:]
    # ffill = on reporte le dernier prix connu (passé uniquement), limité à max_gap jours
    prices = prices.ffill(limit=max_gap).dropna()
    report.update({
        "debut_commun": str(common_start.date()),
        "fin": str(prices.index[-1].date()),
        "nb_jours": int(len(prices)),
        "nb_actifs": int(prices.shape[1]),
    })
    return prices, report


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().iloc[1:]


def align_macro(macro: pd.DataFrame, index: pd.DatetimeIndex, lag: int = 1) -> pd.DataFrame:
    """Aligne la macro sur les jours de bourse, avec un décalage de `lag` jours."""
    full = macro.reindex(index.union(macro.index)).ffill()
    return full.reindex(index).shift(lag)


# ---------------------------------------------------------------- découpage
def split_dates(index: pd.DatetimeIndex, cfg: dict) -> dict[str, pd.DatetimeIndex]:
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    return {
        "train": index[index <= train_end],
        "val": index[(index > train_end) & (index <= val_end)],
        "test": index[index > val_end],
    }


def purge(
    splits: dict[str, pd.DatetimeIndex],
    horizon: int
) -> dict[str, pd.DatetimeIndex]:

    if horizon <= 0:
        raise ValueError("Horizon must be positive.")

    if len(splits["train"]) <= horizon:
        raise ValueError("Insufficient training observations.")

    if len(splits["val"]) <= horizon:
        raise ValueError("Insufficient validation observations.")

    return {
        "train": splits["train"][:-horizon],
        "val": splits["val"][:-horizon],

        # The final unavailable targets are already removed
        # by features.dropna().
        "test": splits["test"],
    }


# ---------------------------------------------------------------- features
def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss)



def make_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    benchmark: str,
    horizon: int
) -> pd.DataFrame:
    """
    Tableau long : une ligne par (date, actif).

    Les features disponibles à la date t utilisent uniquement
    les prix connus jusqu'à t-1.

    La cible représente le rendement logarithmique des
    horizon prochaines séances.
    """
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark absent des prix : {benchmark}")

    if horizon <= 0:
        raise ValueError("Horizon must be positive.")

    logp = np.log(prices)
    ret = logp.diff()

    assets = [c for c in prices.columns if c != benchmark]

    # 1. Calcul des indicateurs à partir des prix historiques
    per_asset = {
        "mom_21": logp - logp.shift(21),
        "mom_63": logp - logp.shift(63),
        "mom_126": logp - logp.shift(126),
        "vol_21": ret.rolling(21).std() * np.sqrt(252),
        "vol_63": ret.rolling(63).std() * np.sqrt(252),
        "rsi_14": rsi(prices),
        "drawdown_252": prices / prices.rolling(252).max() - 1,
    }

    # 2. Décalage : les features à t utilisent les informations
    # disponibles au plus tard à t-1.
    per_asset = {
        name: frame.shift(1)
        for name, frame in per_asset.items()
    }

    # 3. Cible future : rendement logarithmique sur horizon séances
    target = logp.shift(-horizon) - logp

    # 4. Construction du panel par actif
    frames = []

    for asset in assets:
        df = pd.DataFrame({
            name: frame[asset]
            for name, frame in per_asset.items()
        })

        df["target"] = target[asset]
        df["ticker"] = asset
        frames.append(df)

    panel = pd.concat(frames)

    # 5. Features de marché
    market = pd.DataFrame({
        "mkt_mom_21": per_asset["mom_21"][benchmark],
        "mkt_vol_21": per_asset["vol_21"][benchmark],
    })

    # 6. Variables macroéconomiques
    macro_feats = macro.copy()

    if "vix" in macro_feats.columns:
        macro_feats["vix_chg_21"] = (
            macro_feats["vix"] - macro_feats["vix"].shift(21)
        )

    # 7. Fusion des variables
    panel = panel.join(market).join(macro_feats)

    panel.index.name = "date"

    return (
        panel.reset_index()
        .set_index(["date", "ticker"])
        .sort_index()
    )



# ---------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    out = ROOT / cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    horizon = cfg["features"]["horizon"]
    benchmark = cfg["universe"]["benchmark"]

    prices_raw, macro_raw = load_raw(cfg)
    prices, report = clean_prices(prices_raw)
    returns = log_returns(prices)
    macro = align_macro(macro_raw, prices.index, cfg["features"]["macro_lag_days"])

    features = make_features(prices, macro, benchmark, horizon).dropna()
    splits = purge(split_dates(prices.index, cfg), horizon)

    prices.to_csv(out / "prices_clean.csv")
    returns.to_csv(out / "returns.csv")
    macro.to_csv(out / "macro_aligned.csv")
    dates = features.index.get_level_values("date")
    for name, idx in splits.items():
        features[dates.isin(idx)].to_csv(out / f"features_{name}.csv")

    report["features"] = list(features.columns.drop("target"))
    report["split"] = {
        name: {"debut": str(features[dates.isin(idx)].index[0][0].date()),
               "fin": str(features[dates.isin(idx)].index[-1][0].date()),
               "lignes": int(dates.isin(idx).sum())}
        for name, idx in splits.items()
    }
    with open(out / "preparation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Période commune : {report['debut_commun']} -> {report['fin']} "
          f"({report['nb_jours']} jours, {report['nb_actifs']} actifs)")
    for name, s in report["split"].items():
        print(f"  {name:5s} : {s['debut']} -> {s['fin']}  ({s['lignes']} lignes)")
    print(f"Fichiers écrits dans {out}")


if __name__ == "__main__":
    main()
