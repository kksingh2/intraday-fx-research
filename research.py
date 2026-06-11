import pandas as pd, numpy as np, os

DL = "."
df = pd.read_csv(os.path.join(DL, "eurusd_5m_london.csv"), parse_dates=["datetime"])
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
PIP = 0.0001
SPREAD = 0.6 * PIP

ASIAN = (0, 7); ENTRY = (7, 10)
MAX_REENTRY = 2; STOP_BUF = 2 * PIP

daily_close = df.groupby("date")["close"].last()
dates = list(daily_close.index)

def setups():
    """Yield one dict per day describing the sweep+reversal entry (the strategy's core)."""
    for day, g in df.groupby("date"):
        a = g[(g.hour >= ASIAN[0]) & (g.hour < ASIAN[1])]
        if a.empty: continue
        hi, lo = a.high.max(), a.low.min()
        rng = (hi - lo) / PIP
        w = g[(g.hour >= ENTRY[0]) & (g.hour < ENTRY[1])].reset_index(drop=True)
        sh = sl = False; sidx = None; sext = None
        for i in range(len(w)):
            c = w.iloc[i]
            if not sh and c.high > hi: sh, sidx, sext = True, i, c.high
            if not sl and c.low  < lo: sl, sidx, sext = True, i, c.low
            if sh and sidx is not None and i - sidx <= MAX_REENTRY and c.close < hi:
                yield dict(day=day, dir="short", entry=c.close, sweep=sext,
                           hi=hi, lo=lo, rng=rng, t=c.datetime, g=g); break
            if sl and sidx is not None and i - sidx <= MAX_REENTRY and c.close > lo:
                yield dict(day=day, dir="long", entry=c.close, sweep=sext,
                           hi=hi, lo=lo, rng=rng, t=c.datetime, g=g); break
            if sh and sidx is not None and i - sidx > MAX_REENTRY: sh=False; sidx=None
            if sl and sidx is not None and i - sidx > MAX_REENTRY: sl=False; sidx=None

ALL = list(setups())
print(f"Total raw sweep+reversal setups: {len(ALL)}")

# ---------- DIAGNOSTIC: where does price go after entry? ----------
# stop = beyond sweep wick (reversal stop). Measure how far price travels in R-multiples
# in the trade's favour before the stop is hit, intraday.
def mfe_in_R(s):
    if s["dir"] == "short":
        stop = s["sweep"] + STOP_BUF; risk = stop - s["entry"]
    else:
        stop = s["sweep"] - STOP_BUF; risk = s["entry"] - stop
    if risk <= 0: return None
    rest = s["g"][s["g"].datetime > s["t"]]
    best = 0.0
    for _, fc in rest.iterrows():
        if s["dir"] == "short":
            if fc.high >= stop: break
            best = max(best, (s["entry"] - fc.low) / risk)
        else:
            if fc.low <= stop: break
            best = max(best, (fc.high - s["entry"]) / risk)
    return best

mfes = [m for m in (mfe_in_R(s) for s in ALL) if m is not None]
mfes = np.array(mfes)
print("\nDIAGNOSTIC - how far reversals actually run before stop (R, max favourable excursion):")
for thr in [0.5, 1, 1.5, 2, 3, 4]:
    print(f"  reached >= {thr}R before stop : {(mfes >= thr).mean()*100:5.1f}%")
print(f"  median MFE: {np.median(mfes):.2f}R   mean: {mfes.mean():.2f}R")

# ---------- generalized simulator ----------
def simulate(s, mode, target_R, htf=True, htf_n=5, rng_band=(15,80)):
    if not (rng_band[0] <= s["rng"] <= rng_band[1]): return None
    # direction actually traded
    d = s["dir"] if mode == "reversal" else ("long" if s["dir"]=="short" else "short")
    if htf:
        try: i = dates.index(s["day"])
        except ValueError: return None
        if i - htf_n < 0: return None
        bias = "long" if daily_close.iloc[i-1] > daily_close.iloc[i-htf_n] else "short"
        if d != bias: return None
    entry = s["entry"]
    if mode == "reversal":
        stop = s["sweep"] + STOP_BUF if d=="short" else s["sweep"] - STOP_BUF
    else:  # continuation: stop on opposite side of range
        stop = s["hi"] + STOP_BUF if d=="long" else s["lo"] - STOP_BUF
    risk = abs(entry - stop)
    if risk <= 0: return None
    if target_R == "range":
        target = s["lo"] if d=="short" else s["hi"]
    else:
        target = entry - target_R*risk if d=="short" else entry + target_R*risk
    rest = s["g"][s["g"].datetime > s["t"]]
    cost = SPREAD
    for _, fc in rest.iterrows():
        if d=="short":
            if fc.high >= stop: return -(risk+cost)/risk
            if fc.low  <= target: return (abs(target-entry)-cost)/risk
        else:
            if fc.low  <= stop: return -(risk+cost)/risk
            if fc.high >= target: return (abs(target-entry)-cost)/risk
    last = rest.close.iloc[-1] if len(rest) else entry
    pnl = (entry-last) if d=="short" else (last-entry)
    return (pnl - cost)/risk

def expectancy(setups_list, **kw):
    rs = [r for r in (simulate(s, **kw) for s in setups_list) if r is not None]
    if not rs: return (0,0,0)
    rs = np.array(rs)
    return (len(rs), (rs>0).mean()*100, rs.mean())

# ---------- TRAIN / TEST split ----------
def yr(s): return pd.Timestamp(s["day"]).year
TRAIN = [s for s in ALL if yr(s) in (2023,2024)]
TEST  = [s for s in ALL if yr(s) in (2025,2026)]
print(f"\nTRAIN setups (2023-24): {len(TRAIN)}   TEST setups (2025-26): {len(TEST)}")

print("\n" + "="*64)
print("EXPLORE ON TRAIN ONLY (2023-2024) - do not believe until validated")
print("="*64)
print(f"{'mode':<12}{'target':<8}{'htf':<6}{'n':>5}{'win%':>8}{'exp(R)':>9}")
configs = []
for mode in ["reversal","continuation"]:
    for tgt in [1,1.5,2,3,"range"]:
        for htf in [True, False]:
            n,w,e = expectancy(TRAIN, mode=mode, target_R=tgt, htf=htf)
            configs.append((mode,tgt,htf,n,w,e))
            print(f"{mode:<12}{str(tgt):<8}{str(htf):<6}{n:>5}{w:>8.1f}{e:>9.3f}")

# pick the single best by TRAIN expectancy with a sane sample size (>=40 trades)
viable = [c for c in configs if c[3] >= 40]
best = max(viable, key=lambda c: c[5])
print("\nBest on TRAIN (>=40 trades):", best[:3], f" exp={best[5]:.3f}R  n={best[3]}")

print("\n" + "="*64)
print("VALIDATE THAT SINGLE CHOICE ON HELD-OUT TEST (2025-2026)")
print("="*64)
n,w,e = expectancy(TEST, mode=best[0], target_R=best[1], htf=best[2])
print(f"mode={best[0]} target={best[1]} htf={best[2]}")
print(f"  TRAIN: n={best[3]:>4}  win={best[4]:.1f}%  exp={best[5]:+.3f}R")
print(f"  TEST : n={n:>4}  win={w:.1f}%  exp={e:+.3f}R")
print("\n-> If TEST expectancy is not clearly positive, the TRAIN result was overfit.")
