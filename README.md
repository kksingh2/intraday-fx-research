# Intraday FX Research

Backtesting one specific trading idea on the euro/dollar exchange rate, using 5-minute price data. The honest result is the point of the project.

## The idea

During the Asian hours (overnight in London) the price usually stays in a small range. The strategy marks the high and low of that range, then watches the London morning session:

1. Wait for the price to spike just past the Asian high or low. The thinking is that big traders push it there to trigger other people's stop orders.
2. When a candle closes back inside the range, take a trade in the opposite direction, betting the spike was a fake-out.
3. Stop loss goes a few pips beyond the spike. Target is the opposite side of the Asian range.

## What I tried

I ran about 150 small variations across the four `research` scripts: different session hours, different stop sizes, day-of-week filters, and time-of-day filters.

## What I found

In the raw price data it looked profitable. But once I added the spread (the small fee paid on every trade, about 0.6 pips for euro/dollar), the edge disappeared. An idea that looks good before costs but not after is not a real edge.

I am keeping the code here so I remember not to chase the same idea again without modelling costs honestly.

## How the code works

Each script loads the price data with pandas, groups it by day, finds the Asian high and low, then steps through the London candles looking for the sweep-and-reverse pattern. It records every trade and adds up the profit, with a switch to include or exclude the spread.

## Run it

Put a euro/dollar 5-minute CSV with columns `datetime, open, high, low, close` in the folder, then run any of the Python files.
