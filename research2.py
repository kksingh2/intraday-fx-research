import pandas as pd, numpy as np, os

DL = "."
df = pd.read_csv(os.path.join(DL, "eurusd_5m_london.csv"), parse_dates=["datetime"])
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
PIP = 0.0001; SPREAD = 0.6*PIP
ASIAN=(0,7); ENTRY=(7,10); MAX_REENTRY=2; STOP_BUF=2*PIP
daily_close = df.groupby("date")["close"].last(); dates=list(daily_close.index)

def setups():
    for day,g in df.groupby("date"):
        a=g[(g.hour>=ASIAN[0])&(g.hour<ASIAN[1])]
        if a.empty: continue
        hi,lo=a.high.max(),a.low.min(); rng=(hi-lo)/PIP
        w=g[(g.hour>=ENTRY[0])&(g.hour<ENTRY[1])].reset_index(drop=True)
        sh=sl=False; sidx=None; sext=None
        for i in range(len(w)):
            c=w.iloc[i]
            if not sh and c.high>hi: sh,sidx,sext=True,i,c.high
            if not sl and c.low<lo:  sl,sidx,sext=True,i,c.low
            if sh and sidx is not None and i-sidx<=MAX_REENTRY and c.close<hi:
                yield dict(day=day,dir="short",entry=c.close,sweep=sext,hi=hi,lo=lo,
                           rng=rng,t=c.datetime,overshoot=(sext-hi)/PIP,g=g); break
            if sl and sidx is not None and i-sidx<=MAX_REENTRY and c.close>lo:
                yield dict(day=day,dir="long",entry=c.close,sweep=sext,hi=hi,lo=lo,
                           rng=rng,t=c.datetime,overshoot=(lo-sext)/PIP,g=g); break
            if sh and sidx is not None and i-sidx>MAX_REENTRY: sh=False;sidx=None
            if sl and sidx is not None and i-sidx>MAX_REENTRY: sl=False;sidx=None

ALL=list(setups())
def yr(s): return pd.Timestamp(s["day"]).year
TRAIN=[s for s in ALL if yr(s) in (2023,2024)]
TEST =[s for s in ALL if yr(s) in (2025,2026)]
RNG_MED = np.median([s["rng"] for s in TRAIN])   # threshold learned on TRAIN only
print(f"TRAIN={len(TRAIN)}  TEST={len(TEST)}  train range median={RNG_MED:.1f} pips")

def sim(s):  # base: reversal, 1R target, HTF filter, range band 15-80
    if not (15<=s["rng"]<=80): return None
    d=s["dir"]
    i=dates.index(s["day"])
    if i-5<0: return None
    bias="long" if daily_close.iloc[i-1]>daily_close.iloc[i-5] else "short"
    if d!=bias: return None
    entry=s["entry"]
    stop=s["sweep"]+STOP_BUF if d=="short" else s["sweep"]-STOP_BUF
    risk=abs(entry-stop)
    if risk<=0: return None
    target=entry-1*risk if d=="short" else entry+1*risk
    rest=s["g"][s["g"].datetime>s["t"]]
    for _,fc in rest.iterrows():
        if d=="short":
            if fc.high>=stop: return -(risk+SPREAD)/risk
            if fc.low<=target: return (abs(target-entry)-SPREAD)/risk
        else:
            if fc.low<=stop: return -(risk+SPREAD)/risk
            if fc.high>=target: return (abs(target-entry)-SPREAD)/risk
    last=rest.close.iloc[-1] if len(rest) else entry
    pnl=(entry-last) if d=="short" else (last-entry)
    return (pnl-SPREAD)/risk

FILTERS = {
    "BASE (no extra filter)": lambda s: True,
    "1 shallow sweep <5p":    lambda s: s["overshoot"] < 5,
    "2 early <=08:30":        lambda s: (s["t"].hour, s["t"].minute) <= (8,30),
    "3 tight range <median":  lambda s: s["rng"] < RNG_MED,
    "4 mid-week Tue-Thu":     lambda s: pd.Timestamp(s["day"]).dayofweek in (1,2,3),
}

def evalf(setlist, filt):
    rs=[sim(s) for s in setlist if filt(s)]
    rs=[r for r in rs if r is not None]
    if not rs: return (0,0,0)
    rs=np.array(rs); return (len(rs),(rs>0).mean()*100,rs.mean())

print("\n"+"="*60+"\nEXPLORE FILTERS ON TRAIN (2023-2024)\n"+"="*60)
print(f"{'filter':<26}{'n':>5}{'win%':>8}{'exp(R)':>9}")
res={}
for name,f in FILTERS.items():
    n,w,e=evalf(TRAIN,f); res[name]=(n,w,e)
    print(f"{name:<26}{n:>5}{w:>8.1f}{e:>9.3f}")

cand={k:v for k,v in res.items() if not k.startswith("BASE") and v[0]>=40}
best=max(cand,key=lambda k:cand[k][2])
print(f"\nBest filter on TRAIN (>=40 trades): {best}  exp={cand[best][2]:+.3f}R")

print("\n"+"="*60+"\nVALIDATE ON HELD-OUT TEST (2025-2026)\n"+"="*60)
n,w,e=evalf(TEST,FILTERS[best])
bn,bw,be=cand[best]
print(f"{best}")
print(f"  TRAIN: n={bn:>4} win={bw:.1f}% exp={be:+.3f}R")
print(f"  TEST : n={n:>4} win={w:.1f}% exp={e:+.3f}R")
print("\nVerdict: profitable out-of-sample only if TEST exp is clearly > 0.")
