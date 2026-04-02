"""ML-based price direction predictor.

Trains a GradientBoostingClassifier on historical indicator features
to predict next-day price direction (up/down). Retrains on each call
with the full available history — no model persistence needed.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 60


def build_feature_matrix(df: pd.DataFrame, indicators: dict) -> pd.DataFrame:
    """Build a feature DataFrame from price data and computed indicators."""
    close = df["Close"].values.astype(float)
    features = pd.DataFrame(index=df.index)

    # RSI
    rsi = indicators.get("rsi")
    if rsi is not None:
        features["rsi"] = rsi.values

    # MACD histogram and signal diff
    macd_data = indicators.get("macd", {})
    if "histogram" in macd_data:
        features["macd_hist"] = macd_data["histogram"].values
    if "macd" in macd_data and "signal" in macd_data:
        features["macd_signal_diff"] = macd_data["macd"].values - macd_data["signal"].values

    # Price relative to moving averages
    ma_data = indicators.get("ma", {})
    if ma_data.get("sma_20") is not None:
        sma20 = ma_data["sma_20"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            features["sma20_ratio"] = close / sma20
    if ma_data.get("sma_50") is not None:
        sma50 = ma_data["sma_50"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            features["sma50_ratio"] = close / sma50

    # Bollinger Band position
    bb_data = indicators.get("bb", {})
    if bb_data.get("upper") is not None and bb_data.get("lower") is not None:
        upper = bb_data["upper"].values
        lower = bb_data["lower"].values
        band_width = upper - lower
        with np.errstate(divide="ignore", invalid="ignore"):
            features["bb_position"] = np.where(
                band_width > 0, (close - lower) / band_width, 0.5
            )

    # OBV slope (10-day normalized)
    obv = indicators.get("obv")
    if obv is not None:
        obv_vals = obv.values.astype(float)
        obv_sma10 = pd.Series(obv_vals).rolling(10).mean().values
        with np.errstate(divide="ignore", invalid="ignore"):
            features["obv_slope_10"] = np.where(
                np.abs(obv_sma10) > 0,
                (obv_vals - obv_sma10) / np.abs(obv_sma10),
                0.0,
            )

    # Stochastic
    stoch_data = indicators.get("stochastic", {})
    if stoch_data.get("slowk") is not None:
        features["stoch_k"] = stoch_data["slowk"].values
    if stoch_data.get("slowd") is not None:
        features["stoch_d"] = stoch_data["slowd"].values

    # ATR normalized by price
    atr = indicators.get("atr")
    if atr is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            features["atr_normalized"] = atr.values / close

    # Volume relative to 20-day average
    volume = indicators.get("volume")
    if volume is not None:
        vol_vals = volume.values.astype(float)
        vol_sma20 = pd.Series(vol_vals).rolling(20).mean().values
        with np.errstate(divide="ignore", invalid="ignore"):
            features["volume_sma_ratio"] = np.where(vol_sma20 > 0, vol_vals / vol_sma20, 1.0)

    # Fibonacci position
    fib = indicators.get("fibonacci")
    if fib is not None:
        features["fib_position"] = fib["current_position"]

    # Target: next day up (1) or down (0)
    features["target"] = (pd.Series(close).shift(-1) > close).astype(float)

    return features


def train_and_predict(df: pd.DataFrame, indicators: dict) -> tuple[float, float, str]:
    """Train a GBM classifier and predict next-day direction.

    Returns:
        (probability, score, detail_string)
        - probability: 0.0-1.0 chance of price going up
        - score: -2.0 to +2.0 mapped from probability
        - detail: human-readable description
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        features = build_feature_matrix(df, indicators)

        # Drop rows with NaN and separate target
        features = features.replace([np.inf, -np.inf], np.nan)
        clean = features.dropna()

        if len(clean) < MIN_TRAINING_SAMPLES:
            return 0.5, 0.0, f"ML: Insufficient data ({len(clean)} samples, need {MIN_TRAINING_SAMPLES})"

        # Split: all except last row for training, last row for prediction
        X = clean.drop(columns=["target"])
        y = clean["target"]

        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        X_pred = X.iloc[-1:].copy()

        if y_train.nunique() < 2:
            return 0.5, 0.0, "ML: Insufficient class diversity"

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42, min_samples_leaf=5
        )
        model.fit(X_train, y_train)

        prob_up = model.predict_proba(X_pred)[0][1]

        # Map probability to score
        if prob_up > 0.70:
            score = 2.0
        elif prob_up > 0.60:
            score = 1.0
        elif prob_up >= 0.40:
            score = 0.0
        elif prob_up >= 0.30:
            score = -1.0
        else:
            score = -2.0

        direction = "Up" if prob_up > 0.5 else "Down"
        confidence = abs(prob_up - 0.5) * 200  # 0-100%

        return (
            prob_up,
            score,
            f"ML: {prob_up:.0%} up probability ({direction}, {confidence:.0f}% confidence)",
        )

    except Exception as e:
        logger.warning(f"ML prediction failed: {e}")
        return 0.5, 0.0, f"ML: Prediction failed"
