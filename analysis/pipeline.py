"""Full analysis pipeline — runs indicators, sentiment, ML, and generates signal + chart.

Shared by Telegram handlers, agent executor, and MCP server to avoid duplication.
"""

import logging
from io import BytesIO

from analysis.indicators import compute_all
from analysis.signals import generate_signal, SignalResult
from analysis.charts import generate_chart
from analysis.sentiment import get_sentiment_score
from analysis.ml_predictor import train_and_predict
from data.market import MarketData

logger = logging.getLogger(__name__)


async def run_full_analysis(
    market: MarketData, ticker: str
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
    chart_buf = generate_chart(ticker, df, indicators, signal)

    return signal, chart_buf, indicators
