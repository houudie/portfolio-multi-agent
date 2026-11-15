"""Téléchargement des données ouvertes (étape 2).

- Prix journaliers ajustés : Yahoo Finance via yfinance
- Variables macro : FRED (CSV public, sans clé d'API)

Usage : python -m src.data.download
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import ROOT, load_config

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return data["Close"].sort_index()


def download_fred(series: dict[str, str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for code, name in series.items():
        df = pd.read_csv(FRED_URL.format(series=code), index_col=0, parse_dates=True)
        df.columns = [name]
        frames.append(pd.to_numeric(df[name], errors="coerce").to_frame())
    macro = pd.concat(frames, axis=1)
    return macro.loc[start:end]


def main() -> None:
    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)

    u = cfg["universe"]
    tickers = u["stocks"] + u["etfs"] + [u["benchmark"]]
    start, end = cfg["dates"]["start"], cfg["dates"]["end"]

    prices = download_prices(tickers, start, end)
    prices.to_csv(raw / "prices.csv")
    print(f"Prix : {prices.shape[0]} jours x {prices.shape[1]} actifs -> {raw / 'prices.csv'}")

    macro = download_fred(cfg["macro_fred"], start, end)
    macro.to_csv(raw / "macro.csv")
    print(f"Macro : {macro.shape[0]} lignes x {macro.shape[1]} séries -> {raw / 'macro.csv'}")


if __name__ == "__main__":
    main()
