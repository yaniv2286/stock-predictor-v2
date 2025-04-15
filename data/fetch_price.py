import yfinance as yf
import pandas as pd

def fetch_price_data(ticker: str, interval: str = "1d", start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch historical price data using yfinance.
    Normalize all timestamps to tz-naive (UTC).
    """
    try:
        df = yf.download(ticker, interval=interval, start=start, end=end, progress=False,auto_adjust=False)

        df.dropna(inplace=True)
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        # Normalize datetime index
        df.index = df.index.tz_convert(None) if df.index.tz else df.index
        df.index.name = "datetime"
        return df
    except Exception as e:
        print(f"[fetch_price_data] Error fetching {ticker} - {e}")
        return pd.DataFrame()
