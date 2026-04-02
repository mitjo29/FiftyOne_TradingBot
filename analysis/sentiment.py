"""News sentiment analysis using Yahoo Finance headlines.

No API key required — uses yfinance's built-in news feed.
Applies keyword-based scoring to headlines for a sentiment signal.
"""

import asyncio
import logging

import yfinance as yf

logger = logging.getLogger(__name__)

BULLISH_WORDS = {
    "upgrade": 1.5, "upgrades": 1.5, "upgraded": 1.5,
    "beat": 1.0, "beats": 1.0, "beating": 1.0,
    "surge": 1.5, "surges": 1.5, "surging": 1.5,
    "rally": 1.0, "rallies": 1.0,
    "growth": 0.8, "grows": 0.8, "growing": 0.8,
    "profit": 0.8, "profits": 0.8, "profitable": 0.8,
    "outperform": 1.0, "outperforms": 1.0,
    "buy": 0.7, "bullish": 1.2,
    "record": 0.8, "high": 0.5,
    "strong": 0.5, "positive": 0.5,
    "rise": 0.5, "rises": 0.5, "rising": 0.5,
    "gain": 0.5, "gains": 0.5,
    "boost": 0.7, "boosts": 0.7,
    "soar": 1.2, "soars": 1.2,
    "jump": 0.8, "jumps": 0.8,
    "exceed": 0.8, "exceeds": 0.8,
    "optimistic": 0.7, "momentum": 0.5,
}

BEARISH_WORDS = {
    "downgrade": -1.5, "downgrades": -1.5, "downgraded": -1.5,
    "miss": -1.0, "misses": -1.0, "missed": -1.0,
    "plunge": -1.5, "plunges": -1.5, "plunging": -1.5,
    "crash": -1.5, "crashes": -1.5,
    "loss": -0.8, "losses": -0.8,
    "underperform": -1.0, "underperforms": -1.0,
    "sell": -0.7, "bearish": -1.2,
    "warning": -0.8, "warns": -0.8,
    "weak": -0.5, "weakness": -0.5,
    "negative": -0.5,
    "fall": -0.5, "falls": -0.5, "falling": -0.5,
    "decline": -0.7, "declines": -0.7, "declining": -0.7,
    "risk": -0.3, "risks": -0.3, "risky": -0.5,
    "cut": -0.5, "cuts": -0.5,
    "drop": -0.7, "drops": -0.7, "dropping": -0.7,
    "slump": -1.0, "slumps": -1.0,
    "layoff": -0.8, "layoffs": -0.8,
    "lawsuit": -0.7, "investigation": -0.6,
    "pessimistic": -0.7, "concern": -0.4, "concerns": -0.4,
}


def score_headline(headline: str) -> float:
    words = headline.lower().split()
    score = 0.0
    for word in words:
        # Strip common punctuation
        clean = word.strip(".,;:!?\"'()-")
        if clean in BULLISH_WORDS:
            score += BULLISH_WORDS[clean]
        elif clean in BEARISH_WORDS:
            score += BEARISH_WORDS[clean]
    return max(-2.0, min(2.0, score))


def _fetch_headlines(ticker: str, max_items: int = 10) -> list[str]:
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news:
            return []
        headlines = []
        for item in news[:max_items]:
            title = item.get("title", "")
            if title:
                headlines.append(title)
        return headlines
    except Exception as e:
        logger.warning(f"Failed to fetch news for {ticker}: {e}")
        return []


async def get_sentiment_score(ticker: str) -> tuple[float, str]:
    """Fetch news headlines and return (score, detail_string).

    Score range: -2.0 (very bearish) to +2.0 (very bullish).
    Returns (0.0, "No recent news") if headlines are unavailable.
    """
    headlines = await asyncio.to_thread(_fetch_headlines, ticker.upper())

    if not headlines:
        return 0.0, "Sentiment: No recent news"

    scores = [score_headline(h) for h in headlines]
    avg = sum(scores) / len(scores)

    # Map average to -2..+2 (amplify slightly since individual headlines are often mild)
    final_score = max(-2.0, min(2.0, avg * 1.5))

    if final_score > 0.5:
        mood = "Bullish"
    elif final_score > 0:
        mood = "Slightly bullish"
    elif final_score > -0.5:
        mood = "Neutral"
    elif final_score > -1.0:
        mood = "Slightly bearish"
    else:
        mood = "Bearish"

    return final_score, f"Sentiment: {mood} ({len(headlines)} headlines, avg={avg:.2f})"
