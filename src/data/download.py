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


def download_prices(
    tickers: list[str],
    start: str,
    end: str
) -> pd.DataFrame:

    # yfinance uses an exclusive end date.
    # Our project configuration uses an inclusive end date.
    end_exclusive = (
        pd.Timestamp(end) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    data = yf.download(
        tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
        group_by="column",
        timeout=30,
    )

    if data is None or data.empty:
        raise RuntimeError("Yahoo Finance returned an empty dataset.")

    if not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError(
            "Unexpected Yahoo Finance output format."
        )

    if "Close" not in data.columns.get_level_values(0):
        raise RuntimeError("Adjusted Close prices are unavailable.")

    prices = data["Close"].copy()

    missing = [
        ticker for ticker in tickers
        if ticker not in prices.columns
        or prices[ticker].notna().sum() == 0
    ]

    if missing:
        raise RuntimeError(
            f"Missing or empty tickers: {missing}"
        )

    prices = prices.reindex(columns=tickers).sort_index()

    return prices


def download_fred(series: dict[str, str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for code, name in series.items():
        df = pd.read_csv(FRED_URL.format(series=code), index_col=0, parse_dates=True)
        df.columns = [name]
        frames.append(pd.to_numeric(df[name], errors="coerce").to_frame())
    macro = pd.concat(frames, axis=1, sort=True)
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
