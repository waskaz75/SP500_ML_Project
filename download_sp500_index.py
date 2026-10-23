from pathlib import Path

import yfinance as yf


DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUTPUT_PATH = DATA_DIR / "sp500_index.csv"
START_DATE = "2000-01-03"
# yfinance treats the end date as exclusive. This includes 2026-10-21.
END_DATE = "2026-10-22"


def main() -> None:
    index_data = yf.download(
        "^GSPC",
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if index_data.empty:
        raise RuntimeError("No S&P 500 index data was downloaded.")

    index_data = index_data.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    # Use the conventional index label in the saved project dataset.
    index_data["symbol"] = "SPX"
    index_data = index_data[
        ["date", "open", "high", "low", "close", "volume", "symbol"]
    ]
    price_columns = ["open", "high", "low", "close"]
    index_data[price_columns] = index_data[price_columns].round(2)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index_data.to_csv(OUTPUT_PATH, index=False, float_format="%.2f")

    print(f"Saved {len(index_data):,} rows to {OUTPUT_PATH}")
    print(f"Date range: {index_data['date'].min()} to {index_data['date'].max()}")


if __name__ == "__main__":
    main()
