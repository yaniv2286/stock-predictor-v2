import time
import yfinance as yf
import pandas as pd

def safe_download(ticker, start, end, interval="1d", retries=5):
    import time
    import yfinance as yf

    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if not df.empty:
                return df
            else:
                print(f"[WARN] Empty dataframe for {ticker} (attempt {attempt+1})")
        except Exception as e:
            print(f"[WARN] Attempt {attempt+1} failed: {e}")
        time.sleep(10)  # wait longer between retries
    raise ValueError(f"Failed to fetch data for {ticker} after {retries} retries")


def fetch_price_data(ticker: str, interval: str = "1d", start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch historical price data using yfinance.
    Normalize all timestamps to tz-naive (UTC).
    """
    try:
        df = safe_download(ticker, start=start, end=end, interval=interval)

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
