from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class SignalResult:
    ticker: str
    overall_signal: str
    score: float
    current_price: float
    details: dict
    timestamp: datetime


# Weights for each indicator
WEIGHTS = {
    "rsi": 0.25,
    "macd": 0.30,
    "ma": 0.25,
    "bb": 0.20,
}


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
    mid = bb_data["mid"]

    if upper is None or upper.empty or pd.isna(upper.iloc[-1]):
        return 0.0, "BB: N/A"

    upper_val = upper.iloc[-1]
    lower_val = lower.iloc[-1]
    mid_val = mid.iloc[-1]
    band_width = upper_val - lower_val

    if band_width == 0:
        return 0.0, "BB: No spread"

    position = (current_price - lower_val) / band_width

    if position <= 0.0:
        score = 2.0
        note = "Below lower band - Strong buy"
    elif position <= 0.2:
        score = 1.0
        note = "Near lower band - Buy zone"
    elif position <= 0.4:
        score = 0.5
        note = "Lower half - Slight buy"
    elif position <= 0.6:
        score = 0.0
        note = "Middle of bands - Neutral"
    elif position <= 0.8:
        score = -0.5
        note = "Upper half - Slight sell"
    elif position <= 1.0:
        score = -1.0
        note = "Near upper band - Sell zone"
    else:
        score = -2.0
        note = "Above upper band - Strong sell"

    return score, f"BB: {note} ({position:.0%})"


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


def generate_signal(ticker: str, df: pd.DataFrame, indicators: dict) -> SignalResult:
    current_price = float(df["Close"].iloc[-1])

    rsi_score, rsi_detail = _score_rsi(
        float(indicators["rsi"].iloc[-1]) if indicators["rsi"] is not None and not indicators["rsi"].empty else float("nan")
    )
    macd_score, macd_detail = _score_macd(indicators["macd"])
    ma_score, ma_detail = _score_ma(indicators["ma"], current_price)
    bb_score, bb_detail = _score_bb(indicators["bb"], current_price)

    weighted_score = (
        rsi_score * WEIGHTS["rsi"]
        + macd_score * WEIGHTS["macd"]
        + ma_score * WEIGHTS["ma"]
        + bb_score * WEIGHTS["bb"]
    )

    return SignalResult(
        ticker=ticker.upper(),
        overall_signal=_score_to_signal(weighted_score),
        score=round(weighted_score, 2),
        current_price=current_price,
        details={
            "rsi": {"score": rsi_score, "detail": rsi_detail},
            "macd": {"score": macd_score, "detail": macd_detail},
            "ma": {"score": ma_score, "detail": ma_detail},
            "bb": {"score": bb_score, "detail": bb_detail},
        },
        timestamp=datetime.now(),
    )
