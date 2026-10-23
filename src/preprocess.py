"""Clean and validate the daily S&P 500 index dataset.

This script preserves the raw input file and writes a separate cleaned CSV plus
a JSON summary of every cleaning check. It intentionally does not remove large
market moves because they may be genuine observations rather than data errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "sp500_index.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "sp500_index_clean.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "results" / "data_cleaning_summary.json"

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "symbol"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["volume"]


def clean_data(input_path: Path) -> tuple[pd.DataFrame, dict]:
    """Return a validated copy of the SPX data and a cleaning summary."""
    df = pd.read_csv(input_path)
    original_rows = len(df)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")

    # Keep only the documented fields and standardize their types.
    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_by_column = {column: int(df[column].isna().sum()) for column in REQUIRED_COLUMNS}
    missing_rows = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    exact_duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    duplicate_date_rows = int(df.duplicated(subset=["date"], keep="last").sum())
    df = df.drop_duplicates(subset=["date"], keep="last").copy()

    invalid_price_mask = (df[PRICE_COLUMNS] <= 0).any(axis=1)
    invalid_volume_mask = df["volume"] < 0
    invalid_ohlc_mask = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    invalid_market_rows = invalid_price_mask | invalid_volume_mask | invalid_ohlc_mask

    invalid_price_rows = int(invalid_price_mask.sum())
    invalid_volume_rows = int(invalid_volume_mask.sum())
    invalid_ohlc_rows = int(invalid_ohlc_mask.sum())
    removed_invalid_rows = int(invalid_market_rows.sum())
    df = df.loc[~invalid_market_rows].copy()

    # Standardize ordering, labels, precision, and volume representation.
    df = df.sort_values("date").reset_index(drop=True)
    df["symbol"] = "SPX"
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].round(2)
    if (df["volume"] % 1 == 0).all():
        df["volume"] = df["volume"].astype("int64")

    # Flag large moves for later EDA, but retain them as meaningful market data.
    daily_return = df["close"].pct_change()
    large_move_rows = int((daily_return.abs() >= 0.05).sum())

    summary = {
        "input_file": input_path.name,
        "input_rows": original_rows,
        "output_rows": len(df),
        "rows_removed": original_rows - len(df),
        "missing_values_by_column_before_cleaning": missing_by_column,
        "rows_with_missing_or_invalid_types_removed": missing_rows,
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "duplicate_date_rows_removed": duplicate_date_rows,
        "invalid_price_rows_found": invalid_price_rows,
        "negative_volume_rows_found": invalid_volume_rows,
        "invalid_ohlc_rows_found": invalid_ohlc_rows,
        "invalid_market_rows_removed": removed_invalid_rows,
        "large_daily_moves_at_or_above_5_percent_retained": large_move_rows,
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "end_date": df["date"].max().strftime("%Y-%m-%d"),
        "symbol": "SPX",
    }
    return df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned, summary = clean_data(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False, date_format="%Y-%m-%d", float_format="%.2f")
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Cleaned dataset saved to: {args.output}")
    print(f"Cleaning summary saved to: {args.summary}")


if __name__ == "__main__":
    main()
