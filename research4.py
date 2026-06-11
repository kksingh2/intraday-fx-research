import pandas as pd, numpy as np, os, random
random.seed(7); np.random.seed(7)
DL="."
df=pd.read_csv(os.path.join(DL,"eurusd_5m_london.csv"),parse_dates=["datetime"])
df["date"]=df["datetime"].dt.date
df["hf"]=df["datetime"].dt.hour+df["datetime"].dt.minute/60.0
df["dow"]=pd.to_datetime(df["datetime"]).dt.dayofweek
PIP=0.0001; SPREAD=0.6*PIP; BUF=2*PIP

# precompute per-day numpy arrays
days=[]
daily_last={}
for d,g in df.groupby("date"):
    g=g.sort_values("datetime")
    days.append(dict(date=d, hf=g.hf.values, o=g.open.values, h=g.high.values,
                     l=g.low.values, c=g.close.values, dow=g.dow.values[0]))
    daily_last[d]=g.close.values[-1]
order=[x["date"] for x in days]
lastvals=np.array([daily_last[d] for d in order])
def bias_of(idx):
    if idx-5<0: return None
    return "long" if lastvals[idx-1]>lastvals[idx-5] else "short"

MIN_RISK=3*PIP   # no sub-pip stops; unfillable in reality
def _clip(r): return max(-2.0, min(6.0, r))
def fwd(o,h,l,c,start,d,entry,stop,target):
    risk=abs(entry-stop)
    if risk<MIN_RISK: return None
    for j in range(start+1,len(c)):
        if d=="long":
            if l[j]<=stop:   return _clip(-(risk+SPREAD)/risk)
            if h[j]>=target: return _clip((abs(target-entry)-SPREAD)/risk)
        else:
            if h[j]>=stop:   return _clip(-(risk+SPREAD)/risk)
            if l[j]<=target: return _clip((abs(target-entry)-SPREAD)/risk)
    last=c[-1]
    pnl=(entry-last) if d=="short" else (last-entry)
    return _clip((pnl-SPREAD)/risk)

def trade(day,didx,p):
    hf,o,h,l,c=day["hf"],day["o"],day["h"],day["l"],day["c"]
    # reference range window
    rmask=(hf>=p["r0"])&(hf<p["r1"])
    if rmask.sum()<3: return None
    rhi=h[rmask].max(); rlo=l[rmask].min(); rng=(rhi-rlo)/PIP
    if not (p["rmin"]<=rng<=p["rmax"]): return None
    wmask=(hf>=p["e0"])&(hf<p["e1"])
    idxs=np.where(wmask)[0]
    if len(idxs)==0: return None
    b=bias_of(didx) if p["htf"] else None
    if p["htf"] and b is None: return None
    op=o[idxs[0]]  # session/window open price (for revert)
    for k in idxs:
        ck=c[k]
        d=None
        if p["kind"]=="break":            # close beyond range edge -> continuation
            if ck>rhi: d="long"
            elif ck<rlo: d="short"
        elif p["kind"]=="reversal":        # poke beyond then close back inside
            if h[k]>rhi and ck<rhi: d="short"
            elif l[k]<rlo and ck>rlo: d="long"
        elif p["kind"]=="revert":          # stretched from window open -> fade
            if ck-op> p["stretch"]*rng*PIP: d="short"
            elif op-ck> p["stretch"]*rng*PIP: d="long"
        elif p["kind"]=="momo":            # 3 rising/falling closes -> continue
            if k>=2 and c[k]>c[k-1]>c[k-2]: d="long"
            elif k>=2 and c[k]<c[k-1]<c[k-2]: d="short"
        if d is None: continue
        if p["flip"]: d="long" if d=="short" else "short"
        if p["htf"] and d!=b: return None
        entry=ck
        if p["stopmode"]=="edge":
            stop=rlo-BUF if d=="long" else rhi+BUF
        else:  # fraction of range
            sd=p["stopfrac"]*rng*PIP
            stop=entry-sd if d=="long" else entry+sd
        risk=abs(entry-stop)
        if risk<=0: return None
        if p["tgt"]=="range":
            target=rhi if d=="long" else rlo
        else:
            target=entry+p["tgt"]*risk if d=="long" else entry-p["tgt"]*risk
        # reject degenerate geometry: stop/target must be on the correct sides of entry
        if d=="long" and not (stop<entry<target):  return None
        if d=="short" and not (target<entry<stop): return None
        return fwd(o,h,l,c,k,d,entry,stop,target)
    return None

def run(p, yrs):
    rs=[]
    for didx,day in enumerate(days):
        if pd.Timestamp(day["date"]).year not in yrs: continue
        if p["dow"] is not None and day["dow"] not in p["dow"]: continue
        r=trade(day,didx,p)
        if r is not None: rs.append(r)
    if len(rs)<30: return None
    rs=np.array(rs); return (len(rs), rs.mean())

# ---- random strategy space ----
WINS=[(0,7,7,10),(0,7,7,12),(7,8,8,11),(7,9,9,12),(13,14,14,18),(12,13,13,17),(8,9,9,13)]
def sample():
    r0,r1,e0,e1=random.choice(WINS)
    return dict(r0=r0,r1=r1,e0=e0,e1=e1,
        kind=random.choice(["break","reversal","revert","momo"]),
        flip=random.random()<0.5,
        htf=random.random()<0.5,
        stopmode=random.choice(["edge","frac"]),
        stopfrac=random.choice([0.3,0.5,0.7,1.0]),
        tgt=random.choice([1,1.5,2,3,"range"]),
        stretch=random.choice([0.8,1.0,1.3]),
        rmin=random.choice([5,10,15]), rmax=random.choice([60,80,120]),
        dow=random.choice([None,(1,2,3),(0,1,2,3,4)]))

N=150
rows=[]
for _ in range(N*3):
    p=sample()
    tr=run(p,(2023,2024)); te=run(p,(2025,2026))
    if tr and te:
        rows.append((tr[1],te[1],tr[0],te[0],p))
    if len(rows)>=N: break

train_e=np.array([r[0] for r in rows]); test_e=np.array([r[1] for r in rows])
print(f"Strategies with >=30 trades both halves: {len(rows)}\n")
print("DISTRIBUTION OF EXPECTANCY (R/trade):")
print(f"  TRAIN: mean={train_e.mean():+.3f}  std={train_e.std():.3f}  %positive={100*(train_e>0).mean():.0f}%")
print(f"  TEST : mean={test_e.mean():+.3f}  std={test_e.std():.3f}  %positive={100*(test_e>0).mean():.0f}%")
print(f"\nCorrelation(train_exp, test_exp) = {np.corrcoef(train_e,test_e)[0,1]:+.3f}")
print("  (a real, learnable edge -> strong positive corr; pure noise -> ~0)")

pos_train=[r for r in rows if r[0]>0]
if pos_train:
    pt_test=np.array([r[1] for r in pos_train])
    print(f"\nOf {len(pos_train)} strategies PROFITABLE on train:")
    print(f"  their mean TEST expectancy = {pt_test.mean():+.3f}R")
    print(f"  fraction still profitable on test = {100*(pt_test>0).mean():.0f}%  (chance ~50%)")

best=max(rows,key=lambda r:r[0])
print(f"\nThe single BEST strategy on TRAIN: train={best[0]:+.3f}R (n={best[2]})  ->  TEST={best[1]:+.3f}R (n={best[3]})")
bestte=max(rows,key=lambda r:r[1])
print(f"The single BEST strategy on TEST : test={bestte[1]:+.3f}R -> but its train was {bestte[0]:+.3f}R")
print(f"\nExpected # 'test-profitable' from pure luck: ~{int(0.5*len(rows))};  actually observed: {(test_e>0).sum()}")
