import numpy as np
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


def compute_obv(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        talib.OBV(df["Close"].values, df["Volume"].values.astype(float)),
        index=df.index,
    )


def compute_stochastic(
    df: pd.DataFrame, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3
) -> dict:
    slowk, slowd = talib.STOCH(
        df["High"].values,
        df["Low"].values,
        df["Close"].values,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=0,
        slowd_period=slowd_period,
        slowd_matype=0,
    )
    return {
        "slowk": pd.Series(slowk, index=df.index),
        "slowd": pd.Series(slowd, index=df.index),
    }


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return pd.Series(
        talib.ATR(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=period),
        index=df.index,
    )


def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> dict:
    recent = df.tail(lookback)
    high = float(recent["High"].max())
    low = float(recent["Low"].min())
    diff = high - low

    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {r: high - diff * r for r in ratios}

    current_price = float(df["Close"].iloc[-1])
    position = (high - current_price) / diff if diff > 0 else 0.5

    return {
        "high": high,
        "low": low,
        "levels": levels,
        "current_position": max(0.0, min(1.0, position)),
    }


def compute_all(df: pd.DataFrame) -> dict:
    return {
        "rsi": compute_rsi(df),
        "macd": compute_macd(df),
        "ma": compute_moving_averages(df),
        "bb": compute_bollinger_bands(df),
        "obv": compute_obv(df),
        "stochastic": compute_stochastic(df),
        "atr": compute_atr(df),
        "fibonacci": compute_fibonacci_levels(df),
        "close": df["Close"],
        "volume": df["Volume"],
    }
