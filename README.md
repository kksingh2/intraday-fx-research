# Intraday FX research

Backtesting an Asian-session liquidity-sweep strategy on EUR/USD 5-minute data, with ~150 variants tried across the four `research*.py` scripts.

## The base strategy

1. Mark the Asian session high and low (00:00-07:00 London)
2. Wait for price to sweep one of those levels during the London window (wick takes the level)
3. Enter when a candle closes back inside the range, against the sweep direction
4. Stop a few pips beyond the sweep wick
5. Target the opposite side of the Asian range
6. Risk a fixed percent per trade

## What I found

No edge after costs. The raw signal had a small positive expectancy in-sample, but it disappeared once spread (0.6 pip) and slippage were modelled honestly. Filters that helped in-sample mostly failed out-of-sample. Logged here as a real "things I tried that didn't work" record.

## Files

- `asian_sweep_backtest.py` - the cleanest single-file backtest, with a CONFIG block to toggle filters
- `research.py` to `research4.py` - exploratory variants (sessions, holding times, re-entry rules, day-of-week filters)

## Running

You'll need your own EUR/USD 5-minute bid CSV with columns `datetime, open, high, low, close`. Drop it in the repo root as `eurusd_5m_london.csv` and run any script with Python 3.10+.
