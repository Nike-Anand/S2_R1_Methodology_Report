"""
Step 4 - Player ratings.
  components (all oriented 'higher = better', per 100 balls unless stated; all are residuals vs situational expectation,
  opposition-adjusted):
     R      scoring / run-prevention value under pressure
     Wsave  dismissal-avoidance (batters) or wicket-taking (bowlers) value under pressure, in runs via lambda_w
     Delta  clutch differential: pressure value minus own normal-situation value
     DOT    dot-ball control (pp vs expected)
     BD     boundary hitting (batters) / boundary prevention (bowlers) (pp vs expected)
     CONS   share of pressure innings/spells with positive net value
  Each component: heteroskedastic empirical-Bayes (DerSimonian-Laird tau^2, match-clustered SEs) -> shrunk talent estimate.
  Composite = sum_k w_k * (theta_k - mu0_k)/tau_k   (talent-SD units; safe as tau->0 because shrinkage factor -> 0)
"""
import json, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
WTS = {"R": 0.25, "Wsave": 0.15, "Delta": 0.20, "DOT": 0.15, "BD": 0.10, "CONS": 0.15}
GATE = dict(nP=100, mP=15, nN=100)
HI, LO = 0.75, 0.50
COMPS = list(WTS)


def row_components(d, side):
    if side == "bat":
        R = 100 * d.r_bat_adj.values; W = 100 * d.lam_w.values * d.out_bat_adj.values
    else:
        R = -100 * d.runs_bowl_adj.values; W = -100 * d.lam_w.values * d.wkt_bowl_adj.values
    return R, W, R - W


def make_table(rows, side, pi_col, hi=HI, lo=LO, mask=None):
    d = rows if mask is None else rows[mask]
    pid = "striker_id" if side == "bat" else "bowler_id"
    R, W, V = row_components(d, side)
    df = pd.DataFrame({"pid": d[pid].values, "mid": d.match_id.values, "R": R, "W": W, "V": V,
                       "DOT": d.dot_pp.values, "BD": d.bd_pp.values, "pi": d[pi_col].values})
    P = df.pi >= hi; N = df.pi <= lo
    gP = df[P].groupby(["pid", "mid"]).agg(nP=("R", "size"), SR=("R", "sum"), SW=("W", "sum"), SVp=("V", "sum"),
                                          SDOT=("DOT", "sum"), SBD=("BD", "sum"))
    gN = df[N].groupby(["pid", "mid"]).agg(nN=("V", "size"), SVn=("V", "sum"), SRn=("R", "sum"), SWn=("W", "sum"), SDOTn=("DOT", "sum"), SBDn=("BD", "sum"))
    T = gP.join(gN, how="outer").fillna(0).reset_index()
    T["flag"] = (T.nP >= 3).astype(float)
    T["cons"] = ((T.SVp > 0) & (T.nP >= 3)).astype(float)
    var = {"R": df.R[P].var(), "W": df.W[P].var(), "Vp": df.V[P].var(), "Vn": df.V[N].var(), "DOT": df.DOT[P].var(), "BD": df.BD[P].var()}
    return T, var


def eligible(T, gate=GATE):
    g = T.groupby("pid").agg(nP=("nP", "sum"), mP=("nP", lambda s: (s > 0).sum()), nN=("nN", "sum"))
    return g[(g.nP >= gate["nP"]) & (g.mP >= gate["mP"]) & (g.nN >= gate["nN"])].index.values


def eb(y, se2):
    se2 = np.maximum(se2, 1e-12)
    w = 1 / se2; k = len(y)
    mu_fe = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - mu_fe) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c)
    wr = 1 / (se2 + tau2); mu0 = np.sum(wr * y) / np.sum(wr)
    Bf = tau2 / (tau2 + se2)
    theta = mu0 + Bf * (y - mu0)
    return theta, Bf, tau2, mu0



def hier(yN, sN, yP, sP):
    """bivariate empirical Bayes: theta_P = theta_N + delta; returns posterior theta_P, posterior delta, hyperparameters."""
    _, _, a, mN = eb(yN, sN)
    _, _, c, md = eb(yP - yN, sP + sN)
    a = max(a, 1e-12)
    det = (a + sN) * (a + c + sP) - a * a
    i1 = yN - mN; i2 = yP - (mN + md)
    g11 = a * (c + sP) / det; g12 = a * sN / det; g21 = -a * c / det; g22 = c * (a + sN) / det
    thN = mN + g11 * i1 + g12 * i2
    dl = md + g21 * i1 + g22 * i2
    return thN + dl, dl, dict(a=a, c=c, mN=mN, md=md)


def cluster_stats(pidx, K, ww, n, S, var_floor):
    nn = np.bincount(pidx, ww * n, K); SS = np.bincount(pidx, ww * S, K)
    mean = np.divide(SS, nn, out=np.zeros(K), where=nn > 0)
    resid = S - mean[pidx] * n
    cs = np.bincount(pidx, ww * resid ** 2, K)
    G = np.bincount(pidx, ww * (n > 0), K)
    se2 = np.divide(cs, nn ** 2, out=np.full(K, 1e6), where=nn > 0) * np.divide(G, np.maximum(G - 1, 1))
    floor = np.divide(0.7 * var_floor, nn, out=np.full(K, 1e6), where=nn > 0)
    return mean, np.maximum(se2, floor), nn


def score(T, var, elig, w_match=None, wts=WTS, return_parts=False, hierarchical=False):
    """returns DataFrame indexed by pid (eligible players) with raw / shrunk components and composite."""
    pmap = pd.Series(np.arange(len(elig)), index=elig)
    Tm = T[T.pid.isin(elig)].copy()
    pidx = Tm.pid.map(pmap).values; K = len(elig)
    mids = np.unique(T.mid.values)
    mi = np.searchsorted(mids, Tm.mid.values)
    ww = np.ones(len(Tm)) if w_match is None else w_match[mi]
    out = {}
    mR, sR, nP = cluster_stats(pidx, K, ww, Tm.nP.values, Tm.SR.values, var["R"])
    mW, sW, _ = cluster_stats(pidx, K, ww, Tm.nP.values, Tm.SW.values, var["W"])
    mDOT, sDOT, _ = cluster_stats(pidx, K, ww, Tm.nP.values, Tm.SDOT.values, var["DOT"])
    mBD, sBD, _ = cluster_stats(pidx, K, ww, Tm.nP.values, Tm.SBD.values, var["BD"])
    mVp, sVp, _ = cluster_stats(pidx, K, ww, Tm.nP.values, Tm.SVp.values, var["Vp"])
    mVn, sVn, nN = cluster_stats(pidx, K, ww, Tm.nN.values, Tm.SVn.values, var["Vn"])
    if hierarchical:
        mRn, sRn, _ = cluster_stats(pidx, K, ww, Tm.nN.values, Tm.SRn.values, var["R"])
        mWn, sWn, _ = cluster_stats(pidx, K, ww, Tm.nN.values, Tm.SWn.values, var["W"])
        mDn, sDn, _ = cluster_stats(pidx, K, ww, Tm.nN.values, Tm.SDOTn.values, var["DOT"])
        mBn, sBn, _ = cluster_stats(pidx, K, ww, Tm.nN.values, Tm.SBDn.values, var["BD"])
    ninn = np.bincount(pidx, ww * Tm.flag.values, K)
    cons = np.divide(np.bincount(pidx, ww * Tm.cons.values, K), ninn, out=np.full(K, 0.5), where=ninn > 0)
    pbar = np.average(cons, weights=np.maximum(ninn, 1e-9))
    sCONS = np.divide(pbar * (1 - pbar), ninn, out=np.full(K, 1e6), where=ninn > 0)
    raw = {"R": (mR, sR), "Wsave": (-mW, sW), "Delta": (mVp - mVn, sVp + sVn), "DOT": (mDOT, sDOT), "BD": (mBD, sBD), "CONS": (cons, sCONS)}
    res = pd.DataFrame(index=elig)
    comp = np.zeros(K); parts = {}
    HN = {"R": (mRn, sRn, mR, sR, 1), "Wsave": (-mWn, sWn, -mW, sW, 1), "DOT": (mDn, sDn, mDOT, sDOT, 1), "BD": (mBn, sBn, mBD, sBD, 1)} if hierarchical else {}
    for k, (y, s2) in raw.items():
        if hierarchical and k in HN:
            yN_, sN_, yP_, sP_, _ = HN[k]
            th, _, hp = hier(yN_, sN_, yP_, sP_)
            tau2 = hp["a"] + hp["c"]; mu0 = hp["mN"] + hp["md"]; Bf = np.ones(K)
        elif hierarchical and k == "Delta":
            _, th, hp = hier(mVn, sVn, mVp, sVp)
            tau2 = hp["c"]; mu0 = hp["md"]; Bf = np.ones(K)
        else:
            th, Bf, tau2, mu0 = eb(y, s2)
        z = (th - mu0) / np.sqrt(tau2) if tau2 > 1e-10 else np.zeros(K)
        res[f"{k}_raw"] = y; res[f"{k}_se"] = np.sqrt(s2); res[f"{k}_shr"] = th; res[f"{k}_B"] = Bf; res[f"{k}_z"] = z
        parts[k] = dict(tau=float(np.sqrt(tau2)), mu0=float(mu0), meanB=float(Bf.mean()))
        comp += wts.get(k, 0) * z
    res["score"] = comp
    res["nP"] = nP; res["nN"] = nN; res["ninn"] = ninn
    res["Value_shr"] = res.R_shr + res.Wsave_shr          # runs above expectation per 100 balls (wicket-adjusted), shrunk
    res["Value_raw"] = res.R_raw + res.Wsave_raw
    res["Delta_raw_se"] = res.Delta_se
    # unshrunk benchmark composite (z of raw using observed SD)  -> to show what shrinkage prevents
    zr = sum(wts.get(k, 0) * (res[f"{k}_raw"] - res[f"{k}_raw"].mean()) / res[f"{k}_raw"].std() for k in COMPS)
    res["score_unshrunk"] = zr
    if return_parts:
        return res, parts
    return res


def bootstrap(T, var, elig, B=300, seed=11, hierarchical=False):
    rng = np.random.default_rng(seed)
    mids = np.unique(T.mid.values); M = len(mids)
    sc = np.zeros((B, len(elig)))
    for b in range(B):
        w = np.bincount(rng.integers(0, M, M), minlength=M).astype(float)
        sc[b] = score(T, var, elig, w_match=w, hierarchical=hierarchical).score.values
    return sc
