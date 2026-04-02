import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from analysis.indicators import compute_all
from analysis.signals import generate_signal
from analysis.charts import generate_chart
from bot.formatters import (
    format_analysis,
    format_portfolio,
    format_overview,
    format_trade,
    format_watchlist,
    format_help,
)
from config import MAX_WATCHLIST_SIZE, VIRTUAL_CASH
from data.market import MarketData
from portfolio.manager import PortfolioManager
from storage.database import Database

logger = logging.getLogger(__name__)


def _get_deps(context: ContextTypes.DEFAULT_TYPE):
    return (
        context.bot_data["db"],
        context.bot_data["market"],
        context.bot_data["portfolio"],
    )


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    await db.ensure_user(update.effective_chat.id)

    has_agent = "agent" in context.bot_data
    ai_note = (
        "\n\n💬 <b>Just chat with me naturally!</b>\n"
        "Try: \"What do you think about Tesla?\" or \"Buy $1000 of Apple\""
        if has_agent
        else ""
    )

    await update.message.reply_text(
        "Welcome to <b>FiftyOne Trading Bot</b>! 📊\n\n"
        "I analyze stocks using technical indicators (RSI, MACD, Moving Averages, "
        "Bollinger Bands) and give you buy/sell/hold signals.\n\n"
        f"Get started with /analyze AAPL or type /help for all commands.{ai_note}",
        parse_mode=ParseMode.HTML,
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(format_help(), parse_mode=ParseMode.HTML)


async def analyze_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db, market, _ = _get_deps(context)
    await db.ensure_user(update.effective_chat.id)

    if not context.args:
        await update.message.reply_text("Usage: /analyze <TICKER>\nExample: /analyze AAPL")
        return

    ticker = context.args[0].upper()
    status_msg = await update.message.reply_text(f"Analyzing {ticker}... ⏳")

    try:
        df = await market.get_stock_data(ticker)
        indicators = compute_all(df)
        signal = generate_signal(ticker, df, indicators)

        chart_buf = generate_chart(ticker, df, indicators, signal)
        text = format_analysis(signal)

        await update.message.reply_photo(photo=chart_buf, caption=text, parse_mode=ParseMode.HTML)
        await status_msg.delete()
    except ValueError as e:
        await status_msg.edit_text(f"Error: {e}")
    except Exception as e:
        logger.exception(f"Error analyzing {ticker}")
        await status_msg.edit_text(f"Failed to analyze {ticker}. Please try again.")


async def watchlist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user_id = update.effective_chat.id
    await db.ensure_user(user_id)

    tickers = await db.get_watchlist(user_id)
    await update.message.reply_text(format_watchlist(tickers), parse_mode=ParseMode.HTML)


async def watch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db, market, _ = _get_deps(context)
    user_id = update.effective_chat.id
    await db.ensure_user(user_id)

    if not context.args:
        await update.message.reply_text("Usage: /watch <TICKER>\nExample: /watch TSLA")
        return

    ticker = context.args[0].upper()

    count = await db.get_watchlist_count(user_id)
    if count >= MAX_WATCHLIST_SIZE:
        await update.message.reply_text(
            f"Watchlist full ({MAX_WATCHLIST_SIZE} max). Remove a stock with /unwatch first."
        )
        return

    valid = await market.validate_ticker(ticker)
    if not valid:
        await update.message.reply_text(f"Ticker '{ticker}' not found.")
        return

    added = await db.add_to_watchlist(user_id, ticker)
    if added:
        await update.message.reply_text(f"Added <b>{ticker}</b> to your watchlist.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(f"{ticker} is already in your watchlist.")


async def unwatch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text("Usage: /unwatch <TICKER>\nExample: /unwatch TSLA")
        return

    ticker = context.args[0].upper()
    removed = await db.remove_from_watchlist(user_id, ticker)
    if removed:
        await update.message.reply_text(f"Removed <b>{ticker}</b> from your watchlist.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(f"{ticker} is not in your watchlist.")


async def portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db, _, portfolio_mgr = _get_deps(context)
    user_id = update.effective_chat.id
    await db.ensure_user(user_id)

    try:
        summary = await portfolio_mgr.get_summary(user_id)
        await update.message.reply_text(format_portfolio(summary), parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.exception("Error fetching portfolio")
        await update.message.reply_text("Failed to load portfolio. Please try again.")


async def overview_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db, _, portfolio_mgr = _get_deps(context)
    user_id = update.effective_chat.id
    await db.ensure_user(user_id)

    try:
        summary = await portfolio_mgr.get_summary(user_id)
        trades = await db.get_trades(user_id)
        text = format_overview(summary, trades, VIRTUAL_CASH)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    except Exception:
        logger.exception("Error fetching overview")
        await update.message.reply_text("Failed to load overview. Please try again.")


async def dashboard_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db, _, portfolio_mgr = _get_deps(context)
    user_id = update.effective_chat.id
    await db.ensure_user(user_id)

    status_msg = await update.message.reply_text("Generating dashboard... ⏳")

    try:
        summary = await portfolio_mgr.get_summary(user_id)
        snapshots = await db.get_snapshots(user_id)

        from portfolio.charts import generate_dashboard
        chart_buf = generate_dashboard(summary, snapshots, VIRTUAL_CASH)

        # Build caption
        total_return = summary.total_value - VIRTUAL_CASH
        ret_sign = "+" if total_return >= 0 else ""
        ret_pct = (total_return / VIRTUAL_CASH * 100) if VIRTUAL_CASH > 0 else 0
        caption = (
            f"<b>Portfolio Dashboard</b>\n"
            f"Value: ${summary.total_value:,.2f} | "
            f"Return: {ret_sign}${total_return:,.2f} ({ret_sign}{ret_pct:.1f}%)\n"
            f"Positions: {len(summary.positions)} | Cash: ${summary.cash:,.2f}"
        )

        await update.message.reply_photo(
            photo=chart_buf, caption=caption, parse_mode=ParseMode.HTML
        )
        await status_msg.delete()
    except Exception:
        logger.exception("Error generating dashboard")
        await status_msg.edit_text("Failed to generate dashboard. Please try again.")


async def buy_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db, _, portfolio_mgr = _get_deps(context)
    user_id = update.effective_chat.id
    await db.ensure_user(user_id)

    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /buy <TICKER> <AMOUNT>\nExample: /buy AAPL 1000"
        )
        return

    ticker = context.args[0].upper()
    try:
        amount = float(context.args[1])
        if amount <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Amount must be a positive number.\nExample: /buy AAPL 1000")
        return

    try:
        trade = await portfolio_mgr.buy(user_id, ticker, amount)
        await update.message.reply_text(format_trade(trade), parse_mode=ParseMode.HTML)
    except ValueError as e:
        await update.message.reply_text(f"Error: {e}")
    except Exception as e:
        logger.exception(f"Error buying {ticker}")
        await update.message.reply_text("Trade failed. Please try again.")


async def sell_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db, _, portfolio_mgr = _get_deps(context)
    user_id = update.effective_chat.id
    await db.ensure_user(user_id)

    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /sell <TICKER> <AMOUNT>\nExample: /sell AAPL 500"
        )
        return

    ticker = context.args[0].upper()
    try:
        amount = float(context.args[1])
        if amount <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Amount must be a positive number.\nExample: /sell AAPL 500")
        return

    try:
        trade = await portfolio_mgr.sell(user_id, ticker, amount)
        await update.message.reply_text(format_trade(trade), parse_mode=ParseMode.HTML)
    except ValueError as e:
        await update.message.reply_text(f"Error: {e}")
    except Exception as e:
        logger.exception(f"Error selling {ticker}")
        await update.message.reply_text("Trade failed. Please try again.")


async def alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user_id = update.effective_chat.id
    await db.ensure_user(user_id)

    settings = await db.get_alert_settings(user_id)
    status = "ON" if settings["enabled"] else "OFF"

    keyboard = [
        [
            InlineKeyboardButton(
                "Turn OFF" if settings["enabled"] else "Turn ON",
                callback_data="toggle_alerts",
            )
        ],
        [
            InlineKeyboardButton("Strong signals only", callback_data="threshold_strong"),
            InlineKeyboardButton("All signals", callback_data="threshold_all"),
        ],
    ]

    await update.message.reply_text(
        f"<b>Alert Settings</b>\n\n"
        f"Status: <b>{status}</b>\n"
        f"Threshold: <b>{settings['threshold']}</b>\n\n"
        f"Alerts scan your watchlist every hour during market hours.",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def alerts_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    db: Database = context.bot_data["db"]
    user_id = query.from_user.id
    settings = await db.get_alert_settings(user_id)

    if query.data == "toggle_alerts":
        new_enabled = not settings["enabled"]
        await db.update_alert_settings(user_id, new_enabled, settings["threshold"])
        status = "ON" if new_enabled else "OFF"
        await query.edit_message_text(f"Alerts turned <b>{status}</b>.", parse_mode=ParseMode.HTML)
    elif query.data == "threshold_strong":
        await db.update_alert_settings(user_id, settings["enabled"], "strong_buy,strong_sell")
        await query.edit_message_text("Threshold set to <b>strong signals only</b>.", parse_mode=ParseMode.HTML)
    elif query.data == "threshold_all":
        await db.update_alert_settings(user_id, settings["enabled"], "all")
        await query.edit_message_text("Threshold set to <b>all signals</b>.", parse_mode=ParseMode.HTML)


async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle free-text messages via the AI agent."""
    db: Database = context.bot_data["db"]
    user_id = update.effective_chat.id
    await db.ensure_user(user_id)

    agent_brain = context.bot_data.get("agent")
    if not agent_brain:
        await update.message.reply_text(
            "AI agent is not configured. Set ANTHROPIC_API_KEY in your .env file.\n"
            "You can still use slash commands — type /help to see them."
        )
        return

    user_text = update.message.text
    if not user_text:
        return

    # Show typing indicator
    await context.bot.send_chat_action(chat_id=user_id, action="typing")

    try:
        result = await agent_brain.chat(user_id, user_text)

        # Send any charts first
        for chart_buf in result.get("charts", []):
            await update.message.reply_photo(photo=chart_buf)

        # Send text response
        text = result.get("text", "")
        if text:
            # Split long messages for Telegram's 4096 char limit
            while len(text) > 4000:
                split_at = text.rfind("\n", 0, 4000)
                if split_at == -1:
                    split_at = 4000
                await update.message.reply_text(text[:split_at])
                text = text[split_at:].lstrip()
            if text:
                await update.message.reply_text(text)

    except Exception:
        logger.exception("Error in AI agent chat")
        await update.message.reply_text(
            "Sorry, I had trouble processing that. Try again or use /help for commands."
        )


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear the AI conversation history for this user."""
    agent_brain = context.bot_data.get("agent")
    if agent_brain:
        agent_brain.clear_history(update.effective_chat.id)
    await update.message.reply_text("Conversation cleared. Let's start fresh!")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Exception while handling update: {context.error}", exc_info=context.error)
