from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


from analysis.recommendation import TradeRecommendation


@dataclass
class SignalResult:
    ticker: str
    overall_signal: str
    score: float
    current_price: float
    details: dict
    timestamp: datetime
    ml_probability: float | None = None
    recommendation: TradeRecommendation | None = None


# Weights for each scored component (sum = 1.0)
WEIGHTS = {
    "rsi": 0.12,
    "macd": 0.15,
    "ma": 0.12,
    "bb": 0.10,
    "volume": 0.10,
    "stochastic": 0.10,
    "sentiment": 0.11,
    "ml": 0.20,
}


# ── Existing indicator scorers ─────────────────────────


def _score_rsi(rsi_value: float) -> tuple[float, str]:
    if pd.isna(rsi_value):
        return 0.0, "N/A"
    if rsi_value < 20:
        return 2.0, f"RSI {rsi_value:.1f} - Strongly oversold"
    elif rsi_value < 30:
        return 1.0, f"RSI {rsi_value:.1f} - Oversold"
    elif rsi_value < 45:
        return 0.5, f"RSI {rsi_value:.1f} - Slightly bullish"
    elif rsi_value <= 55:
        return 0.0, f"RSI {rsi_value:.1f} - Neutral"
    elif rsi_value <= 60:
        return -0.5, f"RSI {rsi_value:.1f} - Slightly bearish"
    elif rsi_value <= 70:
        return -1.0, f"RSI {rsi_value:.1f} - Overbought"
    else:
        return -2.0, f"RSI {rsi_value:.1f} - Strongly overbought"


def _score_macd(macd_data: dict) -> tuple[float, str]:
    macd_line = macd_data["macd"]
    histogram = macd_data["histogram"]
    signal_line = macd_data["signal"]

    if macd_line.empty or pd.isna(macd_line.iloc[-1]):
        return 0.0, "N/A"

    macd_val = macd_line.iloc[-1]
    hist_val = histogram.iloc[-1]
    sig_val = signal_line.iloc[-1]

    hist_prev = histogram.iloc[-2] if len(histogram) > 1 else hist_val
    hist_rising = hist_val > hist_prev

    score = 0.0
    if macd_val > sig_val:
        score += 1.0
        if hist_rising:
            score += 0.5
    else:
        score -= 1.0
        if not hist_rising:
            score -= 0.5

    if hist_val > 0 and hist_rising:
        score = min(score + 0.5, 2.0)
    elif hist_val < 0 and not hist_rising:
        score = max(score - 0.5, -2.0)

    direction = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
    return score, f"MACD {direction} (hist: {hist_val:.3f})"


def _score_ma(ma_data: dict, current_price: float) -> tuple[float, str]:
    sma_50 = ma_data["sma_50"]
    sma_200 = ma_data["sma_200"]

    if sma_50 is None or sma_50.empty or pd.isna(sma_50.iloc[-1]):
        sma_50_val = None
    else:
        sma_50_val = sma_50.iloc[-1]

    if sma_200 is None or sma_200.empty or pd.isna(sma_200.iloc[-1]):
        sma_200_val = None
    else:
        sma_200_val = sma_200.iloc[-1]

    score = 0.0
    notes = []

    if sma_50_val is not None:
        if current_price > sma_50_val:
            score += 0.5
            notes.append("Above SMA50")
        else:
            score -= 0.5
            notes.append("Below SMA50")

    if sma_200_val is not None:
        if current_price > sma_200_val:
            score += 0.5
            notes.append("Above SMA200")
        else:
            score -= 0.5
            notes.append("Below SMA200")

    if sma_50_val is not None and sma_200_val is not None:
        if sma_50_val > sma_200_val:
            score += 1.0
            notes.append("Golden cross")
        else:
            score -= 1.0
            notes.append("Death cross")

    score = max(-2.0, min(2.0, score))
    return score, "MA: " + ", ".join(notes) if notes else "MA: Insufficient data"


def _score_bb(bb_data: dict, current_price: float) -> tuple[float, str]:
    upper = bb_data["upper"]
    lower = bb_data["lower"]

    if upper is None or upper.empty or pd.isna(upper.iloc[-1]):
        return 0.0, "BB: N/A"

    upper_val = upper.iloc[-1]
    lower_val = lower.iloc[-1]
    band_width = upper_val - lower_val

    if band_width == 0:
        return 0.0, "BB: No spread"

    position = (current_price - lower_val) / band_width

    if position <= 0.0:
        score, note = 2.0, "Below lower band - Strong buy"
    elif position <= 0.2:
        score, note = 1.0, "Near lower band - Buy zone"
    elif position <= 0.4:
        score, note = 0.5, "Lower half - Slight buy"
    elif position <= 0.6:
        score, note = 0.0, "Middle of bands - Neutral"
    elif position <= 0.8:
        score, note = -0.5, "Upper half - Slight sell"
    elif position <= 1.0:
        score, note = -1.0, "Near upper band - Sell zone"
    else:
        score, note = -2.0, "Above upper band - Strong sell"

    return score, f"BB: {note} ({position:.0%})"


# ── New indicator scorers ──────────────────────────────


def _score_volume(obv_series: pd.Series, close_series: pd.Series) -> tuple[float, str]:
    """Score OBV by comparing its trend to price trend (confirmation vs divergence)."""
    if obv_series is None or len(obv_series) < 12:
        return 0.0, "Volume: N/A"

    # 10-day slopes
    obv_recent = obv_series.iloc[-10:].values.astype(float)
    close_recent = close_series.iloc[-10:].values.astype(float)

    x = np.arange(10)
    obv_clean = obv_recent[~np.isnan(obv_recent)]
    close_clean = close_recent[~np.isnan(close_recent)]

    if len(obv_clean) < 5 or len(close_clean) < 5:
        return 0.0, "Volume: Insufficient data"

    obv_slope = np.polyfit(x[:len(obv_clean)], obv_clean, 1)[0]
    price_slope = np.polyfit(x[:len(close_clean)], close_clean, 1)[0]

    obv_up = obv_slope > 0
    price_up = price_slope > 0

    if obv_up and price_up:
        score = 1.5
        note = "OBV confirms uptrend"
    elif not obv_up and not price_up:
        score = -1.5
        note = "OBV confirms downtrend"
    elif obv_up and not price_up:
        score = 1.0
        note = "Bullish divergence (OBV rising, price falling)"
    else:
        score = -1.0
        note = "Bearish divergence (OBV falling, price rising)"

    return max(-2.0, min(2.0, score)), f"Volume: {note}"


def _score_stochastic(stoch_data: dict) -> tuple[float, str]:
    """Score Stochastic %K/%D with overbought/oversold crossovers."""
    slowk = stoch_data.get("slowk")
    slowd = stoch_data.get("slowd")

    if slowk is None or slowk.empty or pd.isna(slowk.iloc[-1]):
        return 0.0, "Stoch: N/A"

    k = slowk.iloc[-1]
    d = slowd.iloc[-1] if slowd is not None and not slowd.empty else k

    if k < 20 and k > d:
        score = 2.0
        note = f"%K={k:.0f} crossing up from oversold"
    elif k < 20:
        score = 1.0
        note = f"%K={k:.0f} oversold"
    elif k > 80 and k < d:
        score = -2.0
        note = f"%K={k:.0f} crossing down from overbought"
    elif k > 80:
        score = -1.0
        note = f"%K={k:.0f} overbought"
    elif k > d:
        score = 0.5
        note = f"%K={k:.0f} > %D={d:.0f} bullish momentum"
    elif k < d:
        score = -0.5
        note = f"%K={k:.0f} < %D={d:.0f} bearish momentum"
    else:
        score = 0.0
        note = f"%K={k:.0f} neutral"

    return score, f"Stoch: {note}"


# ── Signal conversion ──────────────────────────────────


def _score_to_signal(score: float) -> str:
    if score >= 1.2:
        return "STRONG BUY"
    elif score >= 0.5:
        return "BUY"
    elif score >= -0.5:
        return "HOLD"
    elif score >= -1.2:
        return "SELL"
    else:
        return "STRONG SELL"


# ── Main signal generator ─────────────────────────────


def generate_signal(
    ticker: str,
    df: pd.DataFrame,
    indicators: dict,
    sentiment_score: tuple[float, str] | None = None,
    ml_score: tuple[float, float, str] | None = None,
) -> SignalResult:
    """Generate a combined signal from all available indicators.

    Args:
        ticker: Stock symbol
        df: OHLCV DataFrame
        indicators: Dict from compute_all()
        sentiment_score: Optional (score, detail) from sentiment analysis
        ml_score: Optional (probability, score, detail) from ML predictor
    """
    current_price = float(df["Close"].iloc[-1])

    # Score all technical indicators
    scores = {}

    rsi = indicators.get("rsi")
    rsi_val = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty and not pd.isna(rsi.iloc[-1]) else float("nan")
    scores["rsi"] = _score_rsi(rsi_val)

    scores["macd"] = _score_macd(indicators.get("macd", {}))
    scores["ma"] = _score_ma(indicators.get("ma", {}), current_price)
    scores["bb"] = _score_bb(indicators.get("bb", {}), current_price)
    scores["volume"] = _score_volume(indicators.get("obv"), indicators.get("close"))
    scores["stochastic"] = _score_stochastic(indicators.get("stochastic", {}))

    # Sentiment
    if sentiment_score is not None:
        scores["sentiment"] = sentiment_score
    else:
        scores["sentiment"] = (0.0, "Sentiment: N/A")

    # ML prediction
    ml_probability = None
    if ml_score is not None:
        ml_probability, ml_s, ml_detail = ml_score
        scores["ml"] = (ml_s, ml_detail)
    else:
        scores["ml"] = (0.0, "ML: N/A")

    # Calculate weighted score with dynamic redistribution
    # If any component returned N/A (score=0 and detail contains "N/A"),
    # redistribute its weight to the others
    active_weights = {}
    for key, weight in WEIGHTS.items():
        detail = scores[key][1]
        if "N/A" in detail or "failed" in detail.lower():
            continue
        active_weights[key] = weight

    # Normalize weights
    total_weight = sum(active_weights.values())
    if total_weight > 0:
        normalized = {k: v / total_weight for k, v in active_weights.items()}
    else:
        normalized = {k: 1.0 / len(WEIGHTS) for k in WEIGHTS}

    weighted_score = sum(
        scores[key][0] * normalized.get(key, 0.0) for key in WEIGHTS
    )

    # Build details dict
    details = {}
    for key in WEIGHTS:
        s, detail = scores[key]
        details[key] = {"score": s, "detail": detail, "weight": normalized.get(key, 0.0)}

    return SignalResult(
        ticker=ticker.upper(),
        overall_signal=_score_to_signal(weighted_score),
        score=round(weighted_score, 2),
        current_price=current_price,
        details=details,
        timestamp=datetime.now(),
        ml_probability=ml_probability,
    )
