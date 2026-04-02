#!/usr/bin/env python3
"""Backtest the trading bot's prediction system against historical data.

Simulates running the signal engine day-by-day over a historical period,
then measures:
- Signal accuracy (did the signal direction match next-day price move?)
- ML prediction accuracy
- Cumulative returns if you followed the signals
- Comparison vs buy-and-hold
- Per-signal-type breakdown (Strong Buy through Strong Sell)

Usage:
    python backtest.py                        # Default: AAPL, 2 years
    python backtest.py TSLA                   # Specific ticker
    python backtest.py AAPL MSFT TSLA GOOG    # Multiple tickers
    python backtest.py --period 5y AAPL       # Custom period
"""

import argparse
import sys
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, ".")
from analysis.indicators import compute_all
from analysis.signals import generate_signal, _score_to_signal
from analysis.ml_predictor import train_and_predict
from analysis.sentiment import score_headline


# ── Backtest Engine ────────────────────────────────────


def fetch_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    print(f"  Fetching {period} of daily data for {ticker}...")
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    print(f"  Got {len(data)} trading days ({data.index[0].date()} to {data.index[-1].date()})")
    return data


def run_backtest(
    ticker: str,
    df: pd.DataFrame,
    lookback: int = 252,
    step: int = 1,
) -> pd.DataFrame:
    """Run the signal engine day-by-day over historical data.

    Args:
        ticker: Stock symbol
        df: Full OHLCV DataFrame
        lookback: Days of history to use for each analysis (default 252 = 1 year)
        step: Step size in days (1 = every day, 5 = weekly)

    Returns:
        DataFrame with columns: date, price, signal, score, ml_prob,
        next_day_return, signal_correct
    """
    results = []
    total_days = len(df)
    eval_start = lookback
    eval_end = total_days - 1  # Need next day for actual return

    num_evals = (eval_end - eval_start) // step
    print(f"  Running {num_evals} evaluations (lookback={lookback}, step={step})...")

    for i in range(eval_start, eval_end, step):
        window = df.iloc[i - lookback : i + 1]

        try:
            indicators = compute_all(window)
            # ML prediction (no sentiment in backtest — it's real-time only)
            ml_result = None
            try:
                ml_result = train_and_predict(window, indicators)
            except Exception:
                pass

            signal = generate_signal(ticker, window, indicators, ml_score=ml_result)

            # Actual next-day return
            current_price = float(df["Close"].iloc[i])
            next_price = float(df["Close"].iloc[i + 1])
            next_day_return = (next_price - current_price) / current_price

            # Was the signal direction correct?
            score = signal.score
            signal_bullish = score > 0
            actual_up = next_day_return > 0
            correct = signal_bullish == actual_up if abs(score) > 0.1 else None  # Skip neutral

            results.append({
                "date": df.index[i],
                "price": current_price,
                "signal": signal.overall_signal,
                "score": score,
                "ml_prob": signal.ml_probability,
                "next_day_return": next_day_return,
                "signal_correct": correct,
            })

        except Exception as e:
            continue

        # Progress indicator
        done = (i - eval_start) // step + 1
        if done % 50 == 0:
            print(f"    {done}/{num_evals} days evaluated...")

    return pd.DataFrame(results)


# ── Analysis ───────────────────────────────────────────


def analyze_results(ticker: str, results: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Analyze backtest results and return metrics."""
    if results.empty:
        return {"ticker": ticker, "error": "No results"}

    # Signal accuracy (excluding neutral/HOLD)
    directional = results.dropna(subset=["signal_correct"])
    accuracy = directional["signal_correct"].mean() * 100 if len(directional) > 0 else 0

    # ML accuracy
    ml_results = results.dropna(subset=["ml_prob"])
    if len(ml_results) > 0:
        ml_correct = ((ml_results["ml_prob"] > 0.5) == (ml_results["next_day_return"] > 0)).mean() * 100
    else:
        ml_correct = 0

    # Signal-following returns (position sizing by score)
    # Score > 0 = long, Score < 0 = short (or cash)
    results["strategy_return"] = results["score"].apply(
        lambda s: min(max(s, -1), 1)  # Clamp to [-1, 1]
    ) * results["next_day_return"]

    cum_strategy = (1 + results["strategy_return"]).cumprod()
    total_strategy_return = (cum_strategy.iloc[-1] - 1) * 100 if len(cum_strategy) > 0 else 0

    # Buy and hold
    start_price = results["price"].iloc[0]
    end_price = results["price"].iloc[-1]
    buy_hold_return = ((end_price - start_price) / start_price) * 100

    # Per-signal breakdown
    signal_breakdown = {}
    for sig in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
        subset = results[results["signal"] == sig]
        if len(subset) > 0:
            avg_return = subset["next_day_return"].mean() * 100
            count = len(subset)
            signal_breakdown[sig] = {"count": count, "avg_next_day_pct": avg_return}

    # Win rate by signal strength
    strong_signals = results[results["score"].abs() >= 1.0]
    if len(strong_signals) > 0:
        strong_correct = strong_signals.dropna(subset=["signal_correct"])
        strong_accuracy = strong_correct["signal_correct"].mean() * 100 if len(strong_correct) > 0 else 0
    else:
        strong_accuracy = 0

    # Sharpe ratio (annualized, assuming 252 trading days)
    if results["strategy_return"].std() > 0:
        sharpe = (results["strategy_return"].mean() / results["strategy_return"].std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    cum_max = cum_strategy.cummax()
    drawdown = (cum_strategy - cum_max) / cum_max
    max_drawdown = drawdown.min() * 100

    return {
        "ticker": ticker,
        "period": f"{results['date'].iloc[0].date()} to {results['date'].iloc[-1].date()}",
        "total_days": len(results),
        "signal_accuracy": accuracy,
        "ml_accuracy": ml_correct,
        "strong_signal_accuracy": strong_accuracy,
        "strategy_return": total_strategy_return,
        "buy_hold_return": buy_hold_return,
        "outperformance": total_strategy_return - buy_hold_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "signal_breakdown": signal_breakdown,
        "cum_strategy": cum_strategy,
        "results_df": results,
    }


def print_report(metrics: dict):
    """Print a formatted backtest report."""
    t = metrics["ticker"]
    print(f"\n{'='*60}")
    print(f"  BACKTEST REPORT: {t}")
    print(f"{'='*60}")
    print(f"  Period: {metrics['period']}")
    print(f"  Trading days evaluated: {metrics['total_days']}")
    print()

    print("  ACCURACY:")
    print(f"    Signal direction accuracy:  {metrics['signal_accuracy']:.1f}%")
    print(f"    ML prediction accuracy:     {metrics['ml_accuracy']:.1f}%")
    print(f"    Strong signal accuracy:     {metrics['strong_signal_accuracy']:.1f}%")
    print()

    print("  RETURNS:")
    ret = metrics['strategy_return']
    bh = metrics['buy_hold_return']
    out = metrics['outperformance']
    print(f"    Strategy return:   {'+' if ret >= 0 else ''}{ret:.1f}%")
    print(f"    Buy & hold return: {'+' if bh >= 0 else ''}{bh:.1f}%")
    print(f"    Outperformance:    {'+' if out >= 0 else ''}{out:.1f}%")
    print(f"    Sharpe ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"    Max drawdown:      {metrics['max_drawdown']:.1f}%")
    print()

    print("  SIGNAL BREAKDOWN:")
    for sig in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
        data = metrics["signal_breakdown"].get(sig)
        if data:
            avg = data["avg_next_day_pct"]
            cnt = data["count"]
            emoji = "+" if avg >= 0 else ""
            print(f"    {sig:12s}: {cnt:4d} signals, avg next-day return: {emoji}{avg:.3f}%")

    print(f"{'='*60}\n")


def generate_backtest_chart(metrics: dict) -> str:
    """Generate a backtest performance chart and save to file."""
    results = metrics["results_df"]
    cum_strategy = metrics["cum_strategy"]
    ticker = metrics["ticker"]

    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.subplots_adjust(hspace=0.15)

    dates = results["date"]

    # Panel 1: Cumulative returns — strategy vs buy-and-hold
    buy_hold_cum = results["price"] / results["price"].iloc[0]
    ax1.plot(dates, cum_strategy.values, color="#42a5f5", linewidth=1.5, label="Strategy")
    ax1.plot(dates, buy_hold_cum.values, color="#9e9e9e", linewidth=1.2, alpha=0.7, label="Buy & Hold")
    ax1.axhline(y=1.0, color="#616161", linestyle="--", linewidth=0.5)
    ax1.fill_between(dates, cum_strategy.values, 1.0, where=cum_strategy.values >= 1.0, alpha=0.1, color="#66bb6a")
    ax1.fill_between(dates, cum_strategy.values, 1.0, where=cum_strategy.values < 1.0, alpha=0.1, color="#ef5350")
    ax1.set_title(
        f"Backtest: {ticker}  |  Strategy: {metrics['strategy_return']:+.1f}%  |  "
        f"Buy&Hold: {metrics['buy_hold_return']:+.1f}%  |  "
        f"Accuracy: {metrics['signal_accuracy']:.0f}%",
        fontsize=13, fontweight="bold",
    )
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.2)

    # Panel 2: Signal scores over time
    colors = ["#00e676" if s > 0 else "#ef5350" if s < 0 else "#ffa726" for s in results["score"]]
    ax2.bar(dates, results["score"], color=colors, alpha=0.7, width=1.5)
    ax2.axhline(y=0, color="#9e9e9e", linewidth=0.5)
    ax2.axhline(y=1.2, color="#66bb6a", linestyle=":", linewidth=0.5, alpha=0.5)
    ax2.axhline(y=-1.2, color="#ef5350", linestyle=":", linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("Signal Score")
    ax2.set_ylim(-2.2, 2.2)
    ax2.grid(alpha=0.2)

    # Panel 3: ML probability over time
    ml_probs = results["ml_prob"].dropna()
    if len(ml_probs) > 0:
        ml_dates = results.loc[ml_probs.index, "date"]
        ax3.plot(ml_dates, ml_probs, color="#ab47bc", linewidth=1.0, alpha=0.8)
        ax3.axhline(y=0.5, color="#9e9e9e", linestyle="--", linewidth=0.5)
        ax3.fill_between(ml_dates, ml_probs, 0.5, where=ml_probs >= 0.5, alpha=0.15, color="#66bb6a")
        ax3.fill_between(ml_dates, ml_probs, 0.5, where=ml_probs < 0.5, alpha=0.15, color="#ef5350")
        ax3.set_ylim(0, 1)
    ax3.set_ylabel("ML Prob (Up)")
    ax3.set_xlabel("Date")
    ax3.grid(alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    filename = f"backtest_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(filename, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return filename


# ── Main ───────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest the FiftyOne trading signal engine")
    parser.add_argument("tickers", nargs="*", default=["AAPL"], help="Stock tickers to test")
    parser.add_argument("--period", default="2y", help="Data period (default: 2y)")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window in days (default: 252)")
    parser.add_argument("--step", type=int, default=1, help="Evaluation step in days (default: 1)")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    all_metrics = []

    for ticker in args.tickers:
        ticker = ticker.upper()
        print(f"\n{'─'*60}")
        print(f"  BACKTESTING: {ticker}")
        print(f"{'─'*60}")

        try:
            df = fetch_data(ticker, args.period)
            results = run_backtest(ticker, df, lookback=args.lookback, step=args.step)

            if results.empty:
                print(f"  No results for {ticker}")
                continue

            metrics = analyze_results(ticker, results, df)
            print_report(metrics)

            if not args.no_chart:
                chart_file = generate_backtest_chart(metrics)
                print(f"  Chart saved: {chart_file}")

            all_metrics.append(metrics)

        except Exception as e:
            print(f"  Error backtesting {ticker}: {e}")

    # Summary if multiple tickers
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print(f"  MULTI-TICKER SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Ticker':<8} {'Accuracy':>10} {'ML Acc':>10} {'Strategy':>10} {'B&H':>10} {'Sharpe':>8}")
        print(f"  {'─'*56}")
        for m in sorted(all_metrics, key=lambda x: x["strategy_return"], reverse=True):
            print(
                f"  {m['ticker']:<8} "
                f"{m['signal_accuracy']:>9.1f}% "
                f"{m['ml_accuracy']:>9.1f}% "
                f"{m['strategy_return']:>+9.1f}% "
                f"{m['buy_hold_return']:>+9.1f}% "
                f"{m['sharpe_ratio']:>7.2f}"
            )
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
