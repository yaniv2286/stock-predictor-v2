from data.fetch_price import fetch_price_data
import pandas as pd

def build_model_input(
    ticker: str,
    timeframes: list = ["1d", "1h"],
    start: str = None,
    end: str = None
) -> pd.DataFrame:
    """
    Fetch price data from multiple timeframes, align, and combine features.
    Currently returns only price data. Will expand to indicators, sentiment, etc.
    """
    df_dict = {}
    for tf in timeframes:
        df = fetch_price_data(ticker, interval=tf, start=start, end=end)
        df.columns = [f"{col}_{tf}" for col in df.columns]
        df_dict[tf] = df

    # Combine and align by time
    combined = pd.concat(df_dict.values(), axis=1, join="outer")
    combined.fillna(method="ffill", inplace=True)
    combined.dropna(inplace=True)
    return combined
