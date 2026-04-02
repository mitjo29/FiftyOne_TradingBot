import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram.ext import ContextTypes

from analysis.indicators import compute_all
from analysis.signals import generate_signal
from bot.formatters import format_alert
from config import MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE
from data.market import MarketData
from portfolio.manager import PortfolioManager
from storage.database import Database
from telegram.constants import ParseMode

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


def is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)
    return market_open <= now <= market_close


async def scan_watchlists(context: ContextTypes.DEFAULT_TYPE):
    if not is_market_hours():
        logger.debug("Outside market hours, skipping scan")
        return

    db: Database = context.bot_data["db"]
    market: MarketData = context.bot_data["market"]

    all_watchlists = await db.get_all_watchlists()
    if not all_watchlists:
        return

    # Collect unique tickers
    all_tickers = set()
    for tickers in all_watchlists.values():
        all_tickers.update(tickers)

    # Analyze each ticker
    signals = {}
    for ticker in all_tickers:
        try:
            df = await market.get_stock_data(ticker, period="3mo")
            indicators = compute_all(df)
            signals[ticker] = generate_signal(ticker, df, indicators)
        except Exception as e:
            logger.warning(f"Failed to analyze {ticker} during scan: {e}")

    # Send alerts to each user
    for user_id, tickers in all_watchlists.items():
        try:
            settings = await db.get_alert_settings(user_id)
            if not settings["enabled"]:
                continue

            threshold = settings["threshold"]

            for ticker in tickers:
                if ticker not in signals:
                    continue

                signal = signals[ticker]
                should_alert = False

                if threshold == "all" and signal.overall_signal != "HOLD":
                    should_alert = True
                elif "strong" in threshold:
                    if signal.overall_signal in ("STRONG BUY", "STRONG SELL"):
                        should_alert = True

                if should_alert:
                    text = format_alert(signal)
                    await context.bot.send_message(
                        chat_id=user_id, text=text, parse_mode=ParseMode.HTML
                    )
        except Exception as e:
            logger.warning(f"Failed to send alerts to user {user_id}: {e}")


async def record_portfolio_snapshots(context: ContextTypes.DEFAULT_TYPE):
    if not is_market_hours():
        return

    db: Database = context.bot_data["db"]
    portfolio_mgr: PortfolioManager = context.bot_data["portfolio"]

    user_ids = await db.get_all_user_ids()
    for user_id in user_ids:
        try:
            summary = await portfolio_mgr.get_summary(user_id)
            positions_value = sum(p.market_value for p in summary.positions)
            await db.record_snapshot(user_id, summary.total_value, summary.cash, positions_value)
        except Exception as e:
            logger.warning(f"Failed to record snapshot for user {user_id}: {e}")
