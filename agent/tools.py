"""Tool definitions for the Claude AI agent.

Each tool maps to existing trading bot functionality. Claude calls these
tools during conversation to execute actions on behalf of the user.
"""

TOOL_DEFINITIONS = [
    {
        "name": "analyze_stock",
        "description": (
            "Run full technical analysis on a stock ticker. Returns RSI, MACD, "
            "Moving Averages, Bollinger Bands indicators and an overall "
            "buy/sell/hold signal with a score from -2 (strong sell) to +2 (strong buy). "
            "Also generates a chart image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. AAPL, TSLA, MSFT)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "buy_stock",
        "description": (
            "Paper-buy a stock with a given dollar amount from the user's virtual portfolio. "
            "Deducts cash and adds shares at current market price."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "dollar_amount": {
                    "type": "number",
                    "description": "Dollar amount to invest (e.g. 1000 for $1,000)",
                },
            },
            "required": ["ticker", "dollar_amount"],
        },
    },
    {
        "name": "sell_stock",
        "description": (
            "Paper-sell a stock position for a given dollar amount. "
            "Sells shares at current market price and adds cash back."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "dollar_amount": {
                    "type": "number",
                    "description": "Dollar amount to sell (e.g. 500 for $500)",
                },
            },
            "required": ["ticker", "dollar_amount"],
        },
    },
    {
        "name": "get_portfolio",
        "description": (
            "Get the user's full portfolio summary including cash balance, "
            "all open positions with current market values, unrealized P&L, "
            "and total portfolio value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_watchlist",
        "description": "Get the user's current stock watchlist.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "add_to_watchlist",
        "description": "Add a stock ticker to the user's watchlist for automatic alerts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to watch",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "remove_from_watchlist",
        "description": "Remove a stock ticker from the user's watchlist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to remove",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current market price of a stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_trade_history",
        "description": (
            "Get the user's recent trade history. Optionally filter by ticker."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Optional ticker to filter trades by",
                }
            },
        },
    },
]
