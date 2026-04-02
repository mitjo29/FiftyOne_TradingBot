"""AI Agent brain — manages conversations with Claude and tool execution.

Each Telegram user gets their own conversation history. When a user sends
a message, Claude processes it, optionally calls tools (analyze, buy, sell,
etc.), and returns a natural language response.
"""

import json
import logging
from collections import defaultdict

import anthropic

from agent.executor import ToolExecutor
from agent.tools import TOOL_DEFINITIONS
from config import ANTHROPIC_API_KEY, VIRTUAL_CASH

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are FiftyOne, a smart and friendly AI trading assistant on Telegram.
You help users analyze stocks, manage their virtual portfolio, and make informed trading decisions.

Key facts:
- Users have a virtual portfolio starting with ${VIRTUAL_CASH:,.0f} in paper money.
- All trades are simulated (paper trading) — no real money involved.
- You have access to real market data via Yahoo Finance (1 year of daily data).
- Your analysis uses 8 weighted components:
  * Technical: RSI, MACD, Moving Averages, Bollinger Bands, OBV Volume, Stochastic Oscillator
  * News Sentiment: keyword-based scoring of recent headlines
  * ML Prediction: GradientBoosting classifier predicting next-day price direction
- Additional context: Fibonacci retracement levels, ATR volatility
- You provide buy/sell/hold signals with scores from -2 (strong sell) to +2 (strong buy).
- ML gives a probability of price going up tomorrow — mention this when relevant.

Your personality:
- Be concise but insightful. Don't over-explain unless asked.
- When users ask about a stock, proactively analyze it with your tools.
- Give clear recommendations but always note this is paper trading, not financial advice.
- Use numbers and data to back up your points.
- If a user wants to trade, confirm the ticker and amount before executing.
- When discussing portfolio performance, be honest about both wins and losses.

Important:
- Always use the analyze_stock tool when users ask about a stock's outlook.
- Use get_portfolio before giving portfolio advice.
- Don't hallucinate prices or data — always use tools to get real numbers.
- Keep responses short for Telegram (under 300 words ideally).
"""

# Max conversation history per user (to control token usage)
MAX_HISTORY_MESSAGES = 30


class AgentBrain:
    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self.client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        # Per-user conversation history: {user_id: [messages]}
        self._history: dict[int, list[dict]] = defaultdict(list)

    def _trim_history(self, user_id: int):
        """Keep conversation history within limits."""
        history = self._history[user_id]
        if len(history) > MAX_HISTORY_MESSAGES:
            # Keep the most recent messages, but always start with a user message
            trimmed = history[-MAX_HISTORY_MESSAGES:]
            # Ensure first message is from user
            while trimmed and trimmed[0]["role"] != "user":
                trimmed.pop(0)
            self._history[user_id] = trimmed

    def clear_history(self, user_id: int):
        """Reset conversation for a user."""
        self._history[user_id] = []

    async def chat(self, user_id: int, user_message: str) -> dict:
        """Process a user message and return agent response.

        Returns dict with:
            - 'text': str — the agent's text response
            - 'charts': list[BytesIO] — any chart images to send
        """
        self._history[user_id].append({
            "role": "user",
            "content": user_message,
        })

        charts = []
        max_tool_rounds = 5  # Prevent infinite tool loops

        for _ in range(max_tool_rounds):
            response = await self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=self._history[user_id],
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Add assistant message with tool use blocks
                self._history[user_id].append({
                    "role": "assistant",
                    "content": response.content,
                })

                # Execute each tool call
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info(
                            f"Agent calling tool: {block.name}({json.dumps(block.input)}) "
                            f"for user {user_id}"
                        )
                        result = await self.executor.execute(
                            block.name, block.input, user_id
                        )

                        # Collect charts if any
                        if "chart" in result:
                            charts.append(result["chart"])

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result["text"],
                        })

                # Add tool results to history
                self._history[user_id].append({
                    "role": "user",
                    "content": tool_results,
                })

                # Continue loop — Claude will process tool results
                continue

            # Claude returned a final text response (stop_reason == "end_turn")
            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

            assistant_text = "\n".join(text_parts)

            self._history[user_id].append({
                "role": "assistant",
                "content": assistant_text,
            })

            self._trim_history(user_id)

            return {"text": assistant_text, "charts": charts}

        # Fallback if max tool rounds exceeded
        return {
            "text": "I got a bit carried away with analysis. Could you try a simpler question?",
            "charts": charts,
        }
