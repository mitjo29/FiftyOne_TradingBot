"""Full analysis pipeline — runs indicators, sentiment, ML, and generates signal + chart.

Shared by Telegram handlers, agent executor, and MCP server to avoid duplication.
"""

import logging
from io import BytesIO

import pandas as pd

from analysis.indicators import compute_all
from analysis.signals import generate_signal, SignalResult
from analysis.charts import generate_chart
from analysis.sentiment import get_sentiment_score
from analysis.ml_predictor import train_and_predict
from analysis.recommendation import generate_recommendation
from data.market import MarketData

logger = logging.getLogger(__name__)


async def run_full_analysis(
    market: MarketData, ticker: str, portfolio_value: float = 100000.0
) -> tuple[SignalResult, BytesIO, dict]:
    """Run the complete analysis pipeline for a stock.

    Returns:
        (signal_result, chart_buffer, indicators_dict)
    """
    ticker = ticker.upper()
    df = await market.get_stock_data(ticker)
    indicators = compute_all(df)

    # Sentiment (async, may fail gracefully)
    try:
        sentiment = await get_sentiment_score(ticker)
    except Exception as e:
        logger.warning(f"Sentiment analysis failed for {ticker}: {e}")
        sentiment = None

    # ML prediction (may fail gracefully)
    try:
        ml_result = train_and_predict(df, indicators)
    except Exception as e:
        logger.warning(f"ML prediction failed for {ticker}: {e}")
        ml_result = None

    signal = generate_signal(ticker, df, indicators, sentiment, ml_result)

    # Generate trade recommendation
    atr = indicators.get("atr")
    atr_value = float(atr.iloc[-1]) if atr is not None and not atr.empty and not pd.isna(atr.iloc[-1]) else None

    signal.recommendation = generate_recommendation(
        signal_score=signal.score,
        signal_name=signal.overall_signal,
        current_price=signal.current_price,
        atr_value=atr_value,
        fibonacci=indicators.get("fibonacci"),
        bb_data=indicators.get("bb"),
        portfolio_value=portfolio_value,
    )

    chart_buf = generate_chart(ticker, df, indicators, signal)

    return signal, chart_buf, indicators
