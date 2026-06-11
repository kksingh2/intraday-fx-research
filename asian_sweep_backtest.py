"""
Asian Range Liquidity Sweep Backtest
=====================================

Tests the strategy:
  - Mark the Asian session high/low (00:00-07:00 London)
  - During the London window, wait for price to SWEEP that high or low (wick takes the level)
  - Enter when a candle CLOSES back inside the range, against the sweep direction
  - Stop a few pips beyond the sweep wick
  - Target the opposite side of the Asian range
  - Risk a fixed % per trade

Toggle each filter in the CONFIG block and re-run to watch expectancy change.

INPUT: a 5-minute EUR/USD (or GBP/USD) CSV with columns: datetime, open, high, low, close
       datetime should be parseable (e.g. 2024-01-02 07:05:00). UTC or London tz both fine,
       just set TIMEZONE_IS_LONDON correctly below.
"""

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# CONFIG  -- this is the part you experiment with
# ----------------------------------------------------------------------
CONFIG = {
    "csv_path": "data.csv",          # <-- point this at your file
    "timezone_is_london": True,      # set False if your timestamps are UTC

    # session definitions (London local hours, 24h)
    "asian_start": 0,                # 00:00
    "asian_end": 7,                  # 07:00  (range built over 00:00-07:00)
    "entry_start": 7,                # 07:00  London open
    "entry_end": 10,                 # 10:00  stop looking for entries after this

    # trade mechanics
    "pip_size": 0.0001,              # EUR/USD pip
    "spread_pips": 0.6,              # round-trip transaction cost (EUR/USD ~0.5-1.0)
    "stop_buffer_pips": 2,           # stop this many pips beyond the sweep wick
    "risk_pct": 1.0,                 # % of equity risked per trade
    "starting_equity": 10000,

    # ---- FILTERS (flip these on/off) ----
    "use_htf_bias": True,            # only trade sweeps that reverse WITH the daily trend
    "htf_lookback_days": 5,          # daily trend = close vs close N days ago
    "require_close_inside": True,    # sweep candle must close beyond, next close back inside
    "max_candles_to_reenter": 2,     # must close back inside within this many candles
    "use_range_filter": True,        # skip abnormally large/small Asian ranges
    "range_min_pips": 15,
    "range_max_pips": 80,
    "one_trade_per_day": True,       # take only the first valid setup each day
}


def load_data(cfg):
    df = pd.read_csv(cfg["csv_path"])
    df.columns = [c.strip().lower() for c in df.columns]
    # accept common alternative column names
    rename = {"time": "datetime", "date": "datetime", "timestamp": "datetime"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["datetime"] = pd.to_datetime(df["datetime"])
    if not cfg["timezone_is_london"]:
        # convert UTC -> London (handles BST/GMT automatically)
        df["datetime"] = (df["datetime"].dt.tz_localize("UTC")
                          .dt.tz_convert("Europe/London").dt.tz_localize(None))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df


def daily_closes(df):
    """Last close of each day, for the HTF bias filter."""
    d = df.groupby("date")["close"].last()
    return d


def backtest(cfg):
    df = load_data(cfg)
    pip = cfg["pip_size"]
    dcloses = daily_closes(df)
    daily_index = list(dcloses.index)

    equity = cfg["starting_equity"]
    trades = []

    for day, day_df in df.groupby("date"):
        asian = day_df[(day_df["hour"] >= cfg["asian_start"]) &
                       (day_df["hour"] < cfg["asian_end"])]
        if asian.empty:
            continue
        a_high = asian["high"].max()
        a_low = asian["low"].min()
        range_pips = (a_high - a_low) / pip

        if cfg["use_range_filter"]:
            if range_pips < cfg["range_min_pips"] or range_pips > cfg["range_max_pips"]:
                continue

        # HTF bias: compare today's prior daily close to N days back
        bias = None
        if cfg["use_htf_bias"]:
            try:
                i = daily_index.index(day)
            except ValueError:
                continue
            if i - cfg["htf_lookback_days"] < 0:
                continue
            prev = dcloses.iloc[i - 1]
            ref = dcloses.iloc[i - cfg["htf_lookback_days"]]
            bias = "long" if prev > ref else "short"

        window = day_df[(day_df["hour"] >= cfg["entry_start"]) &
                        (day_df["hour"] < cfg["entry_end"])].reset_index(drop=True)

        swept_high = swept_low = False
        sweep_idx = None
        sweep_extreme = None
        took_trade = False

        for idx in range(len(window)):
            c = window.iloc[idx]

            # detect a sweep
            if not swept_high and c["high"] > a_high:
                swept_high, sweep_idx, sweep_extreme = True, idx, c["high"]
            if not swept_low and c["low"] < a_low:
                swept_low, sweep_idx, sweep_extreme = True, idx, c["low"]

            # look for the re-entry close after a high sweep -> SHORT
            if swept_high and sweep_idx is not None and idx >= sweep_idx:
                if idx - sweep_idx > cfg["max_candles_to_reenter"]:
                    swept_high = False; sweep_idx = None
                    continue
                if c["close"] < a_high:  # closed back inside
                    if cfg["use_htf_bias"] and bias != "short":
                        swept_high = False; sweep_idx = None
                        continue
                    entry = c["close"]
                    stop = sweep_extreme + cfg["stop_buffer_pips"] * pip
                    target = a_low
                    took_trade = True
                    direction = "short"
                    break

            # re-entry close after a low sweep -> LONG
            if swept_low and sweep_idx is not None and idx >= sweep_idx:
                if idx - sweep_idx > cfg["max_candles_to_reenter"]:
                    swept_low = False; sweep_idx = None
                    continue
                if c["close"] > a_low:
                    if cfg["use_htf_bias"] and bias != "long":
                        swept_low = False; sweep_idx = None
                        continue
                    entry = c["close"]
                    stop = sweep_extreme - cfg["stop_buffer_pips"] * pip
                    target = a_high
                    took_trade = True
                    direction = "long"
                    break

        if not took_trade:
            continue

        # simulate the trade forward through the rest of the day
        risk_dist = abs(entry - stop)
        if risk_dist == 0:
            continue
        rest = day_df[day_df["datetime"] > c["datetime"]]
        outcome = None
        for _, fc in rest.iterrows():
            if direction == "short":
                if fc["high"] >= stop:
                    outcome = "loss"; break
                if fc["low"] <= target:
                    outcome = "win"; break
            else:
                if fc["low"] <= stop:
                    outcome = "loss"; break
                if fc["high"] >= target:
                    outcome = "win"; break
        cost = cfg.get("spread_pips", 0) * pip   # round-trip spread cost in price terms
        if outcome is None:
            # closed at end of day at last price
            last = rest["close"].iloc[-1] if len(rest) else entry
            pnl_dist = (entry - last) if direction == "short" else (last - entry)
            r_mult = (pnl_dist - cost) / risk_dist
        elif outcome == "win":
            reward_dist = abs(target - entry)
            r_mult = (reward_dist - cost) / risk_dist
        else:  # loss
            r_mult = -(risk_dist + cost) / risk_dist

        risk_amount = equity * cfg["risk_pct"] / 100
        pnl = risk_amount * r_mult
        equity += pnl
        trades.append({
            "date": day, "direction": direction, "r_mult": r_mult,
            "outcome": outcome or "eod", "pnl": pnl, "equity": equity,
            "range_pips": range_pips,
        })

    return pd.DataFrame(trades), equity


def report(trades, final_equity, cfg):
    if trades.empty:
        print("No trades triggered. Loosen filters or check your data/timezone.")
        return
    n = len(trades)
    wins = trades[trades["r_mult"] > 0]
    losses = trades[trades["r_mult"] <= 0]
    win_rate = len(wins) / n * 100
    avg_win_r = wins["r_mult"].mean() if len(wins) else 0
    avg_loss_r = losses["r_mult"].mean() if len(losses) else 0
    expectancy_r = trades["r_mult"].mean()

    # max drawdown on the equity curve
    eq = trades["equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()

    print("=" * 50)
    print("ASIAN SWEEP BACKTEST RESULTS")
    print("=" * 50)
    print(f"Filters: HTF bias={cfg['use_htf_bias']}  "
          f"range={cfg['use_range_filter']}  "
          f"one/day={cfg['one_trade_per_day']}")
    print("-" * 50)
    print(f"Trades taken         : {n}")
    print(f"Win rate             : {win_rate:.1f}%")
    print(f"Avg win   (R)        : {avg_win_r:.2f}")
    print(f"Avg loss  (R)        : {avg_loss_r:.2f}")
    print(f"Expectancy (R/trade) : {expectancy_r:.3f}")
    print(f"Starting equity      : {cfg['starting_equity']:,.0f}")
    print(f"Final equity         : {final_equity:,.0f}")
    print(f"Total return         : {(final_equity/cfg['starting_equity']-1)*100:.1f}%")
    print(f"Max drawdown         : {max_dd:.1f}%")
    print("=" * 50)
    print("\nRead expectancy first. Positive = edge, negative = bleed.")
    print("Flip one filter at a time in CONFIG and re-run to see what earns its place.")


if __name__ == "__main__":
    trades, final_equity = backtest(CONFIG)
    report(trades, final_equity, CONFIG)
    trades.to_csv("trade_log.csv", index=False)
    print("\nFull trade-by-trade log saved to trade_log.csv")
