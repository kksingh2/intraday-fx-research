import pandas as pd, numpy as np, os
DL="."
df=pd.read_csv(os.path.join(DL,"eurusd_5m_london.csv"),parse_dates=["datetime"])
df["date"]=df["datetime"].dt.date; df["hour"]=df["datetime"].dt.hour
PIP=0.0001; SPREAD=0.6*PIP; BUF=2*PIP
daily_close=df.groupby("date")["close"].last(); dates=list(daily_close.index)

def bias(day):
    i=dates.index(day)
    if i-5<0: return None
    return "long" if daily_close.iloc[i-1]>daily_close.iloc[i-5] else "short"

def fwd(day_df,d,entry,stop,target,t):
    risk=abs(entry-stop)
    if risk<=0: return None
    rest=day_df[day_df.datetime>t]
    for _,fc in rest.iterrows():
        if d=="long":
            if fc.low<=stop:  return -(risk+SPREAD)/risk
            if fc.high>=target: return (abs(target-entry)-SPREAD)/risk
        else:
            if fc.high>=stop: return -(risk+SPREAD)/risk
            if fc.low<=target: return (abs(target-entry)-SPREAD)/risk
    last=rest.close.iloc[-1] if len(rest) else entry
    pnl=(entry-last) if d=="short" else (last-entry)
    return (pnl-SPREAD)/risk

ASIAN_MED=None  # set on train

# ---- strategy signal generators: return list of R-multiples per setlist ----
def orb(day_df, or_win, ent_win, tR, use_htf):
    o=day_df[(day_df.hour>=or_win[0])&(day_df.hour<or_win[1])]
    if len(o)<3: return None
    oh,ol=o.high.max(),o.low.min()
    w=day_df[(day_df.hour>=ent_win[0])&(day_df.hour<ent_win[1])]
    b=bias(day_df.date.iloc[0]) if use_htf else None
    for _,c in w.iterrows():
        if c.close>oh:
            d="long"
            if use_htf and b!="long": return None
            entry=c.close; stop=ol-BUF; target=entry+tR*(entry-stop)
            return fwd(day_df,d,entry,stop,target,c.datetime)
        if c.close<ol:
            d="short"
            if use_htf and b!="short": return None
            entry=c.close; stop=oh+BUF; target=entry-tR*(stop-entry)
            return fwd(day_df,d,entry,stop,target,c.datetime)
    return None

def coil(day_df, tR):
    a=day_df[day_df.hour<7]
    if a.empty: return None
    hi,lo=a.high.max(),a.low.min(); rng=(hi-lo)/PIP
    if ASIAN_MED is None or rng>=ASIAN_MED: return None   # only tight-coil days
    w=day_df[(day_df.hour>=7)&(day_df.hour<11)]
    for _,c in w.iterrows():
        if c.high>hi:
            entry=c.close; stop=lo-BUF; return fwd(day_df,"long",entry,stop,entry+tR*(entry-stop),c.datetime)
        if c.low<lo:
            entry=c.close; stop=hi+BUF; return fwd(day_df,"short",entry,stop,entry-tR*(stop-entry),c.datetime)
    return None

def stretch(day_df, k):
    a=day_df[day_df.hour<7]
    if a.empty: return None
    rng=a.high.max()-a.low.min()
    op=day_df[day_df.hour>=7]
    if op.empty: return None
    o=op.iloc[0].open
    w=day_df[(day_df.hour>=9)&(day_df.hour<14)]
    for _,c in w.iterrows():
        if c.high-o>k*rng:   # stretched up -> fade short back to open
            entry=c.close; stop=entry+0.5*(entry-o); return fwd(day_df,"short",entry,stop,o,c.datetime)
        if o-c.low>k*rng:    # stretched down -> fade long
            entry=c.close; stop=entry-0.5*(o-entry); return fwd(day_df,"long",entry,stop,o,c.datetime)
    return None

def collect(days, fn, **kw):
    rs=[]
    for day,g in days:
        r=fn(g,**kw)
        if r is not None: rs.append(r)
    if not rs: return (0,0,0)
    rs=np.array(rs); return (len(rs),(rs>0).mean()*100,rs.mean())

g_all=list(df.groupby("date"))
def yr(d): return pd.Timestamp(d).year
TRAIN=[(d,g) for d,g in g_all if yr(d) in (2023,2024)]
TEST =[(d,g) for d,g in g_all if yr(d) in (2025,2026)]
ASIAN_MED=np.median([g[g.hour<7].high.max()-g[g.hour<7].low.min()
                     for d,g in TRAIN if not g[g.hour<7].empty])/PIP

STRATS=[
 ("London ORB 1.5R htf", orb, dict(or_win=(7,8),ent_win=(8,11),tR=1.5,use_htf=True)),
 ("London ORB 2R nohtf", orb, dict(or_win=(7,8),ent_win=(8,11),tR=2,use_htf=False)),
 ("NY ORB 1.5R htf",     orb, dict(or_win=(13,14),ent_win=(14,18),tR=1.5,use_htf=True)),
 ("NY ORB 2R nohtf",     orb, dict(or_win=(13,14),ent_win=(14,18),tR=2,use_htf=False)),
 ("Coil breakout 1.5R",  coil,dict(tR=1.5)),
 ("Coil breakout 2R",    coil,dict(tR=2)),
 ("Stretch-revert k=1.0",stretch,dict(k=1.0)),
 ("Stretch-revert k=1.5",stretch,dict(k=1.5)),
]
print(f"train coil median={ASIAN_MED:.1f}p\n")
print(f"{'strategy':<24}{'TRAIN n/win%/exp':>26}   {'TEST n/win%/exp':>26}")
for name,fn,kw in STRATS:
    n,w,e=collect(TRAIN,fn,**kw); tn,tw,te=collect(TEST,fn,**kw)
    flag=" <== POSITIVE BOTH" if (e>0 and te>0) else ""
    print(f"{name:<24}{n:>5}/{w:>5.1f}/{e:>+6.3f}   {tn:>5}/{tw:>5.1f}/{te:>+6.3f}{flag}")

# ---- strategy 5: hourly seasonality (learn on train, trade on test) ----
print("\nHourly seasonality (mean 1h-forward return per London hour):")
df["ret1h"]=df.groupby("date")["close"].shift(-12)-df["close"]   # +1h (12x5m)
tr=df[df["datetime"].dt.year.isin([2023,2024])]
te=df[df["datetime"].dt.year.isin([2025,2026])]
hourly=tr.groupby("hour")["ret1h"].mean()/PIP
best_long=hourly.idxmax(); best_short=hourly.idxmin()
print(f"  train: best long hour={best_long} ({hourly.max():+.2f}p)  best short hour={best_short} ({hourly.min():+.2f}p)")
def seas_exp(data):
    L=(data[data.hour==best_long]["ret1h"]/PIP-0.6).mean()
    S=(-data[data.hour==best_short]["ret1h"]/PIP-0.6).mean()
    return L,S
trL,trS=seas_exp(tr); teL,teS=seas_exp(te)
print(f"  TRAIN long-hr exp={trL:+.2f}p  short-hr exp={trS:+.2f}p (after 0.6p cost)")
print(f"  TEST  long-hr exp={teL:+.2f}p  short-hr exp={teS:+.2f}p (after 0.6p cost)")
print("\n(positive on BOTH train and test = candidate; anything else = noise/overfit)")
