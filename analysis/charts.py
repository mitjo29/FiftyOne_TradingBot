from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from analysis.signals import SignalResult


SIGNAL_COLORS = {
    "STRONG BUY": "#00e676",
    "BUY": "#66bb6a",
    "HOLD": "#ffa726",
    "SELL": "#ef5350",
    "STRONG SELL": "#d50000",
}


def generate_chart(
    ticker: str,
    df: pd.DataFrame,
    indicators: dict,
    signal_result: SignalResult,
) -> BytesIO:
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.subplots_adjust(hspace=0.05)

    dates = df.index

    # --- Price + Bollinger Bands + Moving Averages ---
    ax1.plot(dates, df["Close"], color="#42a5f5", linewidth=1.5, label="Close")

    bb = indicators["bb"]
    if bb["upper"] is not None and not bb["upper"].empty:
        ax1.fill_between(
            dates, bb["lower"], bb["upper"], alpha=0.1, color="#7e57c2", label="Bollinger Bands"
        )
        ax1.plot(dates, bb["upper"], color="#7e57c2", linewidth=0.5, alpha=0.6)
        ax1.plot(dates, bb["lower"], color="#7e57c2", linewidth=0.5, alpha=0.6)

    ma = indicators["ma"]
    if ma["sma_20"] is not None and not ma["sma_20"].empty:
        ax1.plot(dates, ma["sma_20"], color="#ffeb3b", linewidth=0.8, alpha=0.7, label="SMA 20")
    if ma["sma_50"] is not None and not ma["sma_50"].empty:
        ax1.plot(dates, ma["sma_50"], color="#ff9800", linewidth=0.8, alpha=0.7, label="SMA 50")

    signal_color = SIGNAL_COLORS.get(signal_result.overall_signal, "#ffffff")
    ax1.set_title(
        f"{ticker.upper()} - ${signal_result.current_price:.2f}  |  "
        f"Signal: {signal_result.overall_signal} ({signal_result.score:+.2f})",
        fontsize=14,
        fontweight="bold",
        color=signal_color,
    )
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylabel("Price ($)")
    ax1.grid(alpha=0.2)

    # --- RSI ---
    rsi = indicators["rsi"]
    if rsi is not None and not rsi.empty:
        ax2.plot(dates, rsi, color="#42a5f5", linewidth=1.2)
        ax2.axhline(y=70, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.axhline(y=30, color="#66bb6a", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.axhline(y=50, color="#9e9e9e", linestyle=":", linewidth=0.5, alpha=0.5)
        ax2.fill_between(dates, 70, 100, alpha=0.05, color="#ef5350")
        ax2.fill_between(dates, 0, 30, alpha=0.05, color="#66bb6a")
        ax2.set_ylim(0, 100)

    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf
