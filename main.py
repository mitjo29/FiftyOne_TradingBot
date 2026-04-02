import logging

from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler

from bot.handlers import (
    start_handler,
    help_handler,
    analyze_handler,
    watchlist_handler,
    watch_handler,
    unwatch_handler,
    portfolio_handler,
    overview_handler,
    dashboard_handler,
    buy_handler,
    sell_handler,
    alerts_handler,
    alerts_callback_handler,
    error_handler,
)
from config import TELEGRAM_BOT_TOKEN, LOG_LEVEL, SCAN_INTERVAL_MINUTES
from data.market import MarketData
from portfolio.manager import PortfolioManager
from scheduler.jobs import scan_watchlists, record_portfolio_snapshots
from storage.database import Database


def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, LOG_LEVEL, logging.INFO),
    )
    logger = logging.getLogger(__name__)

    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set. Copy .env.example to .env and add your token.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Shared dependencies
    db = Database()
    market = MarketData()
    portfolio_mgr = PortfolioManager(db, market)

    app.bot_data["db"] = db
    app.bot_data["market"] = market
    app.bot_data["portfolio"] = portfolio_mgr

    # Register command handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("analyze", analyze_handler))
    app.add_handler(CommandHandler("watchlist", watchlist_handler))
    app.add_handler(CommandHandler("watch", watch_handler))
    app.add_handler(CommandHandler("unwatch", unwatch_handler))
    app.add_handler(CommandHandler("portfolio", portfolio_handler))
    app.add_handler(CommandHandler("overview", overview_handler))
    app.add_handler(CommandHandler("dashboard", dashboard_handler))
    app.add_handler(CommandHandler("buy", buy_handler))
    app.add_handler(CommandHandler("sell", sell_handler))
    app.add_handler(CommandHandler("alerts", alerts_handler))
    app.add_handler(CallbackQueryHandler(alerts_callback_handler))
    app.add_error_handler(error_handler)

    # Post-init: setup DB and scheduler
    async def post_init(application):
        await db.init()
        logger.info("Database initialized")

        application.job_queue.run_repeating(
            scan_watchlists,
            interval=SCAN_INTERVAL_MINUTES * 60,
            first=30,
        )
        logger.info(f"Watchlist scanner scheduled every {SCAN_INTERVAL_MINUTES} minutes")

        application.job_queue.run_repeating(
            record_portfolio_snapshots,
            interval=SCAN_INTERVAL_MINUTES * 60,
            first=60,
        )
        logger.info("Portfolio snapshot recorder scheduled")

    app.post_init = post_init

    # Cleanup on shutdown
    async def post_shutdown(application):
        await db.close()

    app.post_shutdown = post_shutdown

    logger.info("Starting FiftyOne Trading Bot...")
    app.run_polling()


if __name__ == "__main__":
    main()
