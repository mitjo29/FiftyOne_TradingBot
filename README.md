# FiftyOne Trading Bot

An AI-powered stock trading assistant that runs on Telegram. Analyzes stocks using 8 technical indicators, news sentiment, and machine learning to generate actionable buy/sell recommendations with exact order types, position sizing, stop-loss, and take-profit levels.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

### Technical Analysis (8 Indicators)
| Indicator | Weight | What it measures |
|-----------|--------|------------------|
| MACD (12/26/9) | 15% | Trend direction & momentum |
| RSI (14) | 12% | Overbought / oversold levels |
| Moving Averages (SMA 20/50/200) | 12% | Trend following, golden/death cross |
| Bollinger Bands (20, 2σ) | 10% | Volatility & mean reversion |
| OBV (On-Balance Volume) | 10% | Volume confirmation / divergence |
| Stochastic (%K/%D 14/3/3) | 10% | Momentum oscillator |
| News Sentiment | 11% | Headline-based bullish/bearish scoring |
| **ML Prediction** | **20%** | GradientBoosting next-day direction forecast |

Plus **Fibonacci Retracement** levels and **ATR** volatility (used for stop-loss and ML features, shown on chart).

### Actionable Trade Recommendations
Every analysis includes a concrete trade plan:
- **Order type**: Market (strong signals), Limit (moderate buys), Stop-Limit (moderate sells)
- **Position size**: 4-15% of portfolio, scaled by confidence + volatility
- **Stop-loss**: 1.5x ATR from entry
- **Take-profit**: 2 targets from Fibonacci support/resistance
- **Risk/reward ratio**: Calculated and flagged

### Telegram Bot
| Command | Description |
|---------|-------------|
| `/analyze AAPL` | Full analysis with 4-panel chart + trade recommendation |
| `/buy AAPL 1000` | Paper buy $1,000 of Apple |
| `/sell AAPL 500` | Paper sell $500 of Apple |
| `/portfolio` | View positions & P&L |
| `/overview` | Detailed portfolio with allocation breakdown |
| `/dashboard` | Visual performance chart |
| `/watchlist` | View tracked stocks |
| `/watch TSLA` | Add to watchlist (auto-alerts hourly) |
| `/unwatch TSLA` | Remove from watchlist |
| `/alerts` | Configure alert settings |
| `/clear` | Reset AI conversation |
| `/help` | Show all commands |

### AI Chat (Claude-Powered)
Just type naturally — no commands needed:
- *"What do you think about Tesla?"* — runs full analysis
- *"Buy $2000 of Apple"* — executes paper trade
- *"How's my portfolio doing?"* — portfolio summary with advice
- *"Compare MSFT and GOOG"* — multi-stock analysis

Requires an [Anthropic API key](https://console.anthropic.com/).

### MCP Server
Plug the trading bot directly into **Claude Desktop**, **Claude Code**, or any MCP client:

**12 Tools** — analyze_stock, buy_stock, sell_stock, get_portfolio, get_watchlist, and more
**2 Resources** — `trading://portfolio/summary`, `trading://watchlist`
**3 Prompts** — stock_analysis, portfolio_review, market_scan

### Backtest Engine
Validate the prediction system against historical data:
```bash
python backtest.py AAPL TSLA MSFT --period 2y --step 5
```
Outputs signal accuracy, ML accuracy, strategy vs buy-and-hold returns, Sharpe ratio, max drawdown, and a 3-panel performance chart.

---

## Architecture

```
FiftyOne_TradingBot/
├── main.py                    # Telegram bot entry point
├── mcp_server.py              # MCP server (stdio + HTTP)
├── backtest.py                # Historical backtester
├── config.py                  # Environment config
│
├── analysis/                  # Core analysis engine
│   ├── indicators.py          # 8 technical indicators (TA-Lib)
│   ├── signals.py             # Weighted scoring → buy/sell/hold
│   ├── sentiment.py           # Yahoo Finance headline sentiment
│   ├── ml_predictor.py        # GradientBoosting price predictor
│   ├── recommendation.py      # Order type, sizing, SL/TP levels
│   ├── charts.py              # 4-panel matplotlib charts
│   └── pipeline.py            # Shared analysis orchestrator
│
├── agent/                     # Claude AI agent
│   ├── brain.py               # Conversation manager + tool loop
│   ├── executor.py            # Tool call → bot module mapping
│   └── tools.py               # 9 tool definitions for Claude
│
├── bot/                       # Telegram interface
│   ├── handlers.py            # Command + chat handlers
│   └── formatters.py          # Message formatting
│
├── data/
│   └── market.py              # Yahoo Finance async wrapper
│
├── portfolio/
│   ├── models.py              # Position, Trade, PortfolioSummary
│   ├── manager.py             # Paper trading engine
│   └── charts.py              # Portfolio dashboard charts
│
├── storage/
│   └── database.py            # Async SQLite (aiosqlite)
│
└── scheduler/
    └── jobs.py                # Watchlist scan + portfolio snapshots
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/mitjo29/FiftyOne_TradingBot.git
cd FiftyOne_TradingBot
pip install -r requirements.txt
```

**Dependencies:**
- `python-telegram-bot[job-queue]` — Async Telegram bot
- `yfinance` — Free stock data (no API key)
- `ta-lib` — Technical indicators ([installation guide](https://github.com/TA-Lib/ta-lib-python#dependencies))
- `scikit-learn` — ML price prediction
- `matplotlib` — Chart generation
- `aiosqlite` — Async SQLite
- `anthropic` — Claude AI agent (optional)
- `mcp` — MCP server (optional)

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:
```env
TELEGRAM_BOT_TOKEN=your_token_here        # Required — get from @BotFather
ANTHROPIC_API_KEY=your_key_here           # Optional — enables AI chat
VIRTUAL_CASH=100000.00                    # Starting paper money
SCAN_INTERVAL_MINUTES=60                  # Watchlist scan frequency
```

### 3. Run

**Telegram Bot:**
```bash
python main.py
```

**MCP Server (for Claude Desktop/Code):**
```bash
python mcp_server.py              # stdio transport
python mcp_server.py --http       # HTTP transport on port 8080
```

**Backtest:**
```bash
python backtest.py AAPL --period 2y
python backtest.py AAPL TSLA MSFT GOOG --step 5
```

---

## MCP Setup

### Claude Desktop

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "trading-bot": {
      "command": "python",
      "args": ["/path/to/FiftyOne_TradingBot/mcp_server.py"]
    }
  }
}
```

### Claude Code

Add to `.claude/settings.json`:
```json
{
  "mcpServers": {
    "trading-bot": {
      "command": "python",
      "args": ["/path/to/FiftyOne_TradingBot/mcp_server.py"]
    }
  }
}
```

---

## How the Signal Engine Works

```
Yahoo Finance (1yr daily OHLCV)
    │
    ├─ RSI, MACD, MAs, Bollinger, OBV, Stochastic, ATR, Fibonacci
    │
    ├─ News Sentiment (headline keyword scoring)
    │
    ├─ ML Prediction (GradientBoosting on 11 features)
    │
    └─ Weighted Average → Score (-2.0 to +2.0)
         │
         ├─ ≥ 1.2  → STRONG BUY   (Market order, 15% size)
         ├─ ≥ 0.5  → BUY          (Limit order, 8% size)
         ├─ > -0.5 → HOLD         (Watch levels)
         ├─ ≥ -1.2 → SELL         (Stop-Limit, 8% size)
         └─ < -1.2 → STRONG SELL  (Market order, 15% size)
              │
              └─ Trade Recommendation
                   ├─ Order type + limit price
                   ├─ Position size (ATR-adjusted)
                   ├─ Stop-loss (1.5x ATR)
                   ├─ Take-profit 1 & 2 (Fibonacci)
                   └─ Risk/Reward ratio
```

If sentiment or ML fails, their weight redistributes to the remaining indicators automatically.

---

## Chart Output

Each `/analyze` produces a 4-panel chart:

1. **Price** — Close + Bollinger Bands + SMA 20/50 + Fibonacci retracement levels
2. **RSI** — With overbought (70) / oversold (30) zones
3. **Stochastic** — %K/%D with 80/20 zones
4. **Volume** — Green/red bars + OBV overlay

---

## Disclaimer

This is a **paper trading bot** for educational purposes. All trades are simulated with virtual money. This is **not financial advice**. Do your own research before making real investment decisions.
