"""SPY daily OHLCV loader with yfinance/Tiingo cross-check and split aliasing.

Why this module exists
----------------------
PROPOSAL.html §5 prescribes three sealed-evaluation splits aliased as
`split_A` (train), `split_B` (validation), and `split_C` (sealed). LLM agents
must never see calendar years, named events, or raw OHLCV (§6 invariant
`no_data_snooping`). Two consequences for this module:

  1. The split boundaries are kept in `_SPLIT_BOUNDS`, a private constant.
     Callers ask for `load_split("split_A")` and receive a DataFrame whose
     index is a `RangeIndex` and whose columns are OHLCV only. No dates.
  2. Two independent sources (yfinance, Tiingo) are cross-checked. yfinance is
     fetched with `auto_adjust=False` so we can compare its nominal close
     against Tiingo's nominal close on identical terms; the Parquet cache
     stores back-adjusted OHLCV (for backtest correctness) plus the
     unadjusted close column (for audit).

Tiingo requires a free API token in the `TIINGO_TOKEN` environment variable
(loaded automatically from `.env` at the repo root). Get one at
https://www.tiingo.com/account/api/token .

The Parquet cache at `data/spy_daily.parquet` retains the date index so a
developer can audit the data offline; agents must not read this file directly.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SplitName = Literal["split_A", "split_B", "split_C"]

_SPLIT_BOUNDS: dict[SplitName, tuple[str, str]] = {
    "split_A": ("2010-01-04", "2018-12-31"),
    "split_B": ("2019-01-01", "2020-12-31"),
    "split_C": ("2021-01-01", "2024-12-31"),
}

DATA_DIR = Path("data")
RAW_PARQUET_PATH = DATA_DIR / "spy_daily.parquet"
CROSS_CHECK_TOLERANCE = 0.005

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
PARQUET_COLUMNS = (*OHLCV_COLUMNS, "close_unadjusted")


def _fetch_yfinance() -> pd.DataFrame:
    """Fetch SPY daily bars from yfinance.

    Returns a DataFrame with columns
        open, high, low, close, volume, close_unadjusted
    where the OHLC are back-adjusted for dividends/splits via the
    Adj-Close / Close ratio. SPY has no splits in this window, so the
    factor only reflects dividends.
    """
    end = (pd.Timestamp(_SPLIT_BOUNDS["split_C"][1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        "SPY",
        start=_SPLIT_BOUNDS["split_A"][0],
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned empty data for SPY")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw.index.name = "date"

    factor = raw["adj_close"] / raw["close"]
    out = pd.DataFrame(
        {
            "open": raw["open"] * factor,
            "high": raw["high"] * factor,
            "low": raw["low"] * factor,
            "close": raw["adj_close"],
            "volume": raw["volume"].astype("int64"),
            "close_unadjusted": raw["close"],
        },
        index=raw.index,
    )
    return out


def _fetch_tiingo() -> pd.DataFrame:
    """Fetch SPY daily bars from Tiingo's REST API (requires TIINGO_TOKEN)."""
    token = os.environ.get("TIINGO_TOKEN")
    if not token:
        raise RuntimeError(
            "TIINGO_TOKEN is not set. Put it in .env at the repo root "
            "(`TIINGO_TOKEN=...`) or export it in your shell."
        )
    start = _SPLIT_BOUNDS["split_A"][0]
    end = _SPLIT_BOUNDS["split_C"][1]
    url = (
        "https://api.tiingo.com/tiingo/daily/SPY/prices"
        f"?startDate={start}&endDate={end}&format=json&resampleFreq=daily"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Tiingo request failed: HTTP {e.code} {e.reason}") from e
    if not payload:
        raise RuntimeError(f"Tiingo returned empty data: {url}")
    df = pd.DataFrame(payload)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date").sort_index()
    # Tiingo's `close` is the nominal close; rename to match our schema.
    return df[["open", "high", "low", "close", "volume"]].copy()


def _cross_check(
    yf_unadjusted_close: pd.Series,
    tiingo_df: pd.DataFrame,
    tol: float = CROSS_CHECK_TOLERANCE,
) -> dict:
    """Compare nominal close prices on overlapping dates.

    yfinance's unadjusted close is compared against Tiingo's close. Both
    series are nominal (no dividend back-adjustment), so a small tolerance
    catches feed-level disagreement without flagging methodology drift.
    """
    common = yf_unadjusted_close.index.intersection(tiingo_df.index)
    if len(common) < 100:
        raise ValueError(f"Too few overlapping bars between yfinance and Tiingo: {len(common)}")
    yf_close = yf_unadjusted_close.loc[common]
    ti_close = tiingo_df.loc[common, "close"]
    rel_diff = (yf_close - ti_close).abs() / ti_close
    summary = {
        "shared_bars": int(len(common)),
        "max_rel_diff": float(rel_diff.max()),
        "median_rel_diff": float(rel_diff.median()),
        "fraction_within_tol": float((rel_diff <= tol).mean()),
        "tolerance": tol,
    }
    if summary["max_rel_diff"] > tol:
        worst = rel_diff.sort_values(ascending=False).head(5)
        raise ValueError(
            "yfinance vs Tiingo disagree beyond tolerance: "
            f"max_rel_diff={summary['max_rel_diff']:.4f} tol={tol}. "
            f"Worst: {worst.to_dict()}"
        )
    return summary


def fetch_and_cache(force: bool = False) -> dict:
    """Fetch both sources, cross-check, persist to Parquet."""
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_PARQUET_PATH.exists() and not force:
        logger.info(
            "cache hit at %s; skipping fetch (use force=True to refetch)",
            RAW_PARQUET_PATH,
        )
        cached = pd.read_parquet(RAW_PARQUET_PATH)
        return {"cache": "hit", "bars": int(len(cached))}

    yf_df = _fetch_yfinance()
    tiingo_df = _fetch_tiingo()
    summary = _cross_check(yf_df["close_unadjusted"], tiingo_df)
    yf_df[list(PARQUET_COLUMNS)].to_parquet(RAW_PARQUET_PATH)
    summary.update({"cache": "miss", "bars": int(len(yf_df))})
    return summary


def load_split(name: SplitName) -> pd.DataFrame:
    """Return adjusted OHLCV for the named split with a `RangeIndex` (no dates).

    By default reads `data/spy_daily.parquet`. Setting `LBG_DATA_PATH`
    overrides this — the synthetic-null calibration (PROPOSAL Stage 1.5
    step 4) uses this hook to point at bootstrapped SPY data without
    rewriting the loader interface. The override file must follow the
    same schema as the real cache: a DatetimeIndex named `date` plus
    columns `open / high / low / close / volume / close_unadjusted`.
    """
    if name not in _SPLIT_BOUNDS:
        raise ValueError(f"Unknown split: {name!r}")
    override = os.environ.get("LBG_DATA_PATH")
    parquet_path = Path(override) if override else RAW_PARQUET_PATH
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet cache missing at {parquet_path}. "
            "Run `uv run python -m lbg.data.loader fetch` first, or set "
            "LBG_DATA_PATH to an alternative cache."
        )
    raw = pd.read_parquet(parquet_path)
    start, end = _SPLIT_BOUNDS[name]
    sliced = raw.loc[start:end, list(OHLCV_COLUMNS)].copy()
    return sliced.reset_index(drop=True)


def split_size(name: SplitName) -> int:
    """Number of bars in a split (developer-facing diagnostic)."""
    return int(len(load_split(name)))


def summarize_split(df: pd.DataFrame) -> dict:
    """Developer-facing summary. Returns numeric stats; emits no dates.

    Agents do not call this function; the Orchestrator does not include its
    output in any LLM context.
    """
    import numpy as np

    close = df["close"]
    log_returns = np.log(close / close.shift(1)).dropna()
    return {
        "n_bars": int(len(df)),
        "close_first": float(close.iloc[0]),
        "close_last": float(close.iloc[-1]),
        "close_min": float(close.min()),
        "close_max": float(close.max()),
        "close_mean": float(close.mean()),
        "daily_log_return_mean": float(log_returns.mean()),
        "daily_log_return_std": float(log_returns.std()),
        "nan_cells": int(df.isna().sum().sum()),
        "negative_volume": int((df["volume"] < 0).sum()),
        "non_positive_close": int((df["close"] <= 0).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="lbg.data.loader")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fetch = sub.add_parser("fetch", help="Fetch both sources, cross-check, cache to parquet.")
    p_fetch.add_argument("--force", action="store_true", help="Re-fetch even if cache exists.")
    sub.add_parser("summary", help="Print per-split row counts and price stats.")
    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        summary = fetch_and_cache(force=args.force)
        print(json.dumps(summary, indent=2))
        return 0
    if args.cmd == "summary":
        for split in _SPLIT_BOUNDS:
            df = load_split(split)
            print(split, json.dumps(summarize_split(df)))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
