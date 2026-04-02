import pandas as pd
import talib


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return pd.Series(talib.RSI(df["Close"].values, timeperiod=period), index=df.index)


def compute_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict:
    macd, signal_line, histogram = talib.MACD(
        df["Close"].values, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    return {
        "macd": pd.Series(macd, index=df.index),
        "signal": pd.Series(signal_line, index=df.index),
        "histogram": pd.Series(histogram, index=df.index),
    }


def compute_moving_averages(df: pd.DataFrame) -> dict:
    close = df["Close"].values
    return {
        "sma_20": pd.Series(talib.SMA(close, timeperiod=20), index=df.index),
        "sma_50": pd.Series(talib.SMA(close, timeperiod=50), index=df.index),
        "sma_200": pd.Series(talib.SMA(close, timeperiod=200), index=df.index),
        "ema_12": pd.Series(talib.EMA(close, timeperiod=12), index=df.index),
        "ema_26": pd.Series(talib.EMA(close, timeperiod=26), index=df.index),
    }


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> dict:
    upper, mid, lower = talib.BBANDS(
        df["Close"].values, timeperiod=period, nbdevup=std, nbdevdn=std
    )
    return {
        "upper": pd.Series(upper, index=df.index),
        "mid": pd.Series(mid, index=df.index),
        "lower": pd.Series(lower, index=df.index),
    }


def compute_all(df: pd.DataFrame) -> dict:
    return {
        "rsi": compute_rsi(df),
        "macd": compute_macd(df),
        "ma": compute_moving_averages(df),
        "bb": compute_bollinger_bands(df),
        "close": df["Close"],
    }
