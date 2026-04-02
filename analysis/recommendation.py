"""Trade recommendation engine.

Generates actionable trade instructions from signal analysis:
- Action: BUY / SELL / HOLD
- Order type: Market, Limit, Stop-Limit
- Position size (% of portfolio based on signal strength + ATR risk)
- Entry price / limit price
- Stop-loss (ATR-based)
- Take-profit targets (Fibonacci-based)
- Risk/reward ratio
"""

from dataclasses import dataclass, field


@dataclass
class TradeRecommendation:
    action: str  # "BUY", "SELL", "HOLD"
    confidence: str  # "High", "Medium", "Low"
    order_type: str  # "Market", "Limit", "Stop-Limit"
    entry_price: float  # Recommended entry
    limit_price: float | None  # For limit/stop-limit orders
    stop_loss: float  # Stop-loss price
    take_profit_1: float  # Conservative target
    take_profit_2: float  # Aggressive target
    position_size_pct: float  # % of portfolio to allocate
    position_size_usd: float  # $ amount based on portfolio
    risk_per_share: float  # Entry - stop loss
    reward_per_share: float  # TP1 - entry
    risk_reward_ratio: float  # reward / risk
    notes: list[str] = field(default_factory=list)


def generate_recommendation(
    signal_score: float,
    signal_name: str,
    current_price: float,
    atr_value: float | None,
    fibonacci: dict | None,
    bb_data: dict | None,
    portfolio_value: float = 100000.0,
) -> TradeRecommendation:
    """Generate a full trade recommendation from analysis data.

    Args:
        signal_score: Combined signal score (-2 to +2)
        signal_name: Signal label (e.g. "STRONG BUY")
        current_price: Current stock price
        atr_value: ATR(14) value for volatility-based stops
        fibonacci: Fibonacci levels dict from indicators
        bb_data: Bollinger Bands dict from indicators
        portfolio_value: User's total portfolio value for position sizing
    """
    # ── Determine action ──
    if signal_name in ("STRONG BUY", "BUY"):
        action = "BUY"
    elif signal_name in ("STRONG SELL", "SELL"):
        action = "SELL"
    else:
        action = "HOLD"

    # ── Confidence level ──
    abs_score = abs(signal_score)
    if abs_score >= 1.2:
        confidence = "High"
    elif abs_score >= 0.5:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ── ATR for stop-loss distance ──
    if atr_value is None or atr_value <= 0:
        atr_value = current_price * 0.02  # Fallback: 2% of price

    # ── Stop-loss: 1.5x ATR from entry ──
    atr_stop_distance = atr_value * 1.5

    # ── Fibonacci levels for targets ──
    fib_levels = {}
    if fibonacci and fibonacci.get("levels"):
        fib_levels = fibonacci["levels"]

    # ── Bollinger Bands for limit price reference ──
    bb_lower = None
    bb_upper = None
    if bb_data:
        try:
            bb_lower = float(bb_data["lower"].iloc[-1]) if bb_data.get("lower") is not None else None
            bb_upper = float(bb_data["upper"].iloc[-1]) if bb_data.get("upper") is not None else None
        except (IndexError, TypeError):
            pass

    # ── Build recommendation based on action ──
    notes = []

    if action == "BUY":
        # Entry: for strong signals use market order, otherwise limit below current
        if confidence == "High":
            order_type = "Market"
            entry_price = current_price
            limit_price = None
            notes.append("Strong signal — market order for immediate entry")
        else:
            order_type = "Limit"
            # Set limit at slight discount (0.5 ATR below current, but above BB lower)
            entry_price = current_price
            limit_price = round(current_price - atr_value * 0.5, 2)
            if bb_lower and limit_price < bb_lower:
                limit_price = round(bb_lower, 2)
            notes.append(f"Limit order to enter at better price (${limit_price:.2f})")

        effective_entry = limit_price if limit_price else entry_price

        # Stop-loss below entry
        stop_loss = round(effective_entry - atr_stop_distance, 2)

        # Take-profit targets from Fibonacci levels above current price
        # TP1: nearest Fibonacci resistance above entry (conservative)
        # TP2: next Fibonacci resistance (aggressive)
        tp1, tp2 = _find_buy_targets(effective_entry, fib_levels, atr_value, current_price)

    elif action == "SELL":
        if confidence == "High":
            order_type = "Market"
            entry_price = current_price
            limit_price = None
            notes.append("Strong signal — market order for immediate exit")
        else:
            order_type = "Stop-Limit"
            entry_price = current_price
            # Trigger slightly below current, limit at slight premium
            limit_price = round(current_price + atr_value * 0.3, 2)
            notes.append(f"Stop-limit order (trigger at market, limit ${limit_price:.2f})")

        effective_entry = entry_price

        # Stop-loss above entry (for short / exit protection)
        stop_loss = round(effective_entry + atr_stop_distance, 2)

        # Take-profit targets: Fibonacci support levels below current price
        tp1, tp2 = _find_sell_targets(effective_entry, fib_levels, atr_value, current_price)

    else:
        # HOLD — no trade, but provide reference levels
        return TradeRecommendation(
            action="HOLD",
            confidence=confidence,
            order_type="None",
            entry_price=current_price,
            limit_price=None,
            stop_loss=round(current_price - atr_stop_distance, 2),
            take_profit_1=round(current_price + atr_value * 2, 2),
            take_profit_2=round(current_price + atr_value * 3, 2),
            position_size_pct=0,
            position_size_usd=0,
            risk_per_share=atr_stop_distance,
            reward_per_share=atr_value * 2,
            risk_reward_ratio=round(atr_value * 2 / atr_stop_distance, 2) if atr_stop_distance > 0 else 0,
            notes=["Signal is neutral — no trade recommended", f"Watch for entry at ${current_price - atr_value:.2f} (support) or exit at ${current_price + atr_value:.2f} (resistance)"],
        )

    # ── Position sizing (Kelly-inspired, capped) ──
    # Higher signal strength = larger position, but capped for risk management
    # Max 25% of portfolio per position, scaled by confidence
    if confidence == "High":
        base_pct = 15.0  # 15% for strong signals
    elif confidence == "Medium":
        base_pct = 8.0  # 8% for moderate signals
    else:
        base_pct = 4.0  # 4% for weak signals

    # Adjust by ATR risk: if ATR is high (volatile), reduce size
    atr_pct = (atr_value / current_price) * 100
    if atr_pct > 3.0:
        base_pct *= 0.7
        notes.append("Position reduced — high volatility (ATR)")
    elif atr_pct < 1.0:
        base_pct *= 1.2
        notes.append("Position increased — low volatility")

    position_size_pct = round(min(base_pct, 25.0), 1)
    position_size_usd = round(portfolio_value * position_size_pct / 100, 2)

    # ── Risk/reward ──
    effective_entry = limit_price if limit_price else entry_price
    if action == "BUY":
        risk_per_share = round(effective_entry - stop_loss, 2)
        reward_per_share = round(tp1 - effective_entry, 2)
    else:
        risk_per_share = round(stop_loss - effective_entry, 2)
        reward_per_share = round(effective_entry - tp1, 2)

    rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

    if rr_ratio < 1.0 and action != "HOLD":
        notes.append(f"Caution: risk/reward ratio ({rr_ratio}:1) is below 1:1")
    elif rr_ratio >= 2.0:
        notes.append(f"Favorable risk/reward ({rr_ratio}:1)")

    return TradeRecommendation(
        action=action,
        confidence=confidence,
        order_type=order_type,
        entry_price=round(entry_price, 2),
        limit_price=round(limit_price, 2) if limit_price else None,
        stop_loss=round(stop_loss, 2),
        take_profit_1=round(tp1, 2),
        take_profit_2=round(tp2, 2),
        position_size_pct=position_size_pct,
        position_size_usd=position_size_usd,
        risk_per_share=risk_per_share,
        reward_per_share=reward_per_share,
        risk_reward_ratio=rr_ratio,
        notes=notes,
    )


def _find_buy_targets(
    entry: float, fib_levels: dict, atr: float, current_price: float
) -> tuple[float, float]:
    """Find take-profit levels above entry for a BUY."""
    # Sort Fibonacci levels ascending
    above = sorted([p for p in fib_levels.values() if p > entry])

    if len(above) >= 2:
        tp1 = above[0]
        tp2 = above[1]
    elif len(above) == 1:
        tp1 = above[0]
        tp2 = entry + atr * 4
    else:
        # No Fibonacci levels above — use ATR multiples
        tp1 = entry + atr * 2
        tp2 = entry + atr * 4

    # Ensure minimum distance
    if tp1 - entry < atr * 0.5:
        tp1 = entry + atr * 2
    if tp2 - entry < atr * 1.5:
        tp2 = entry + atr * 4

    return tp1, tp2


def _find_sell_targets(
    entry: float, fib_levels: dict, atr: float, current_price: float
) -> tuple[float, float]:
    """Find take-profit levels below entry for a SELL."""
    below = sorted([p for p in fib_levels.values() if p < entry], reverse=True)

    if len(below) >= 2:
        tp1 = below[0]
        tp2 = below[1]
    elif len(below) == 1:
        tp1 = below[0]
        tp2 = entry - atr * 4
    else:
        tp1 = entry - atr * 2
        tp2 = entry - atr * 4

    if entry - tp1 < atr * 0.5:
        tp1 = entry - atr * 2
    if entry - tp2 < atr * 1.5:
        tp2 = entry - atr * 4

    return tp1, tp2
