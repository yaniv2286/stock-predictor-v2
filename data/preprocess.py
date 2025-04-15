from data.fetch_price import fetch_price_data
from features.indicators import apply_indicators
from features.regime import detect_multi_timeframe_regimes
import pandas as pd
from features.advanced_features import detect_support_resistance
from features.advanced_features import detect_patterns
from features.advanced_features import add_volatility_features
from features.advanced_features import add_drawdown_features
from features.advanced_features import add_price_action_features
from features.advanced_features import add_lag_features
from features.advanced_features import add_time_features
from features.advanced_features import add_momentum_features


def build_model_input(
    ticker: str,
    timeframes: list = ["1d", "1h"],
    start: str = None,
    end: str = None
) -> pd.DataFrame:
    """
    Build a combined input dataframe with multi-timeframe price, indicators, and regimes.
    """
    df_dict = {}       # With renamed columns (for model input)
    raw_price_dict = {}  # With original columns (for regime detection)

    for tf in timeframes:
        df = fetch_price_data(ticker, interval=tf, start=start, end=end)
        if df.empty:
            print(f"[WARN] No data returned for {ticker} at {tf}")
            continue

        raw_price_dict[tf] = df.copy()

        df = apply_indicators(df)
        df = detect_support_resistance(df)
        df = detect_patterns(df)
        df = add_volatility_features(df)
        df = add_momentum_features(df)
        df = add_drawdown_features(df)
        df = add_price_action_features(df)
        df = add_lag_features(df)
        df = add_time_features(df)

        df.columns = [f"{col}_{tf}" for col in df.columns]
        df_dict[tf] = df

    # Merge all timeframes
    combined = pd.concat(df_dict.values(), axis=1, join="outer")
    combined.ffill(inplace=True)
    combined.dropna(inplace=True)

    # Detect regime using original raw price data
    regime_df = detect_multi_timeframe_regimes(raw_price_dict)
    # Align regime to combined's index to remove unmatched NaNs
    regime_df = regime_df.reindex(combined.index)

    combined = combined.join(regime_df)


    return combined
