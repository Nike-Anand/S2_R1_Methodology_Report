"""
Step 2 - Pressure Index.
  mu(b,w)   : expected runs still to be scored from state (balls left b, wickets lost w), monotone GBM trained on innings 1
  sigma     : residual SD model
  WP        : analytic win-probability from (mu, sigma) + Platt calibration, validated with match-grouped CV
  Leverage  : expected |delta WP| over the next-delivery outcome distribution
  PI        : percentile-mapped blend of leverage and perspective-specific stress, scaled by stakes
"""
import json, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
b = pd.read_pickle("balls.pkl")
b["ci"] = (b.ctype == "IPL").astype(int)
b["yi"] = (b.year - 2022).clip(0, 4)
FEATS = ["balls_left", "wkts_before", "ci", "yi"]
LAMBDA_STAKE = 0.20
ALPHA = 0.60          # weight of leverage vs. stress inside PI
diag = {}

# ------------------------------------------------------------------ innings-1 totals and runs-to-go targets
tot1 = b[b.innings == 1].groupby("match_id").score_after.max()
b["final_inn"] = b.groupby(["match_id", "innings"]).score_after.transform("max")
b["ttg"] = b.final_inn - b.score_before


def fit_mu_sigma(train):
    """monotone GBM for mean and for squared residual (cross-fitted residuals for sigma)."""
    X, y = train[FEATS].values, train.ttg.values
    mono = [1, -1, 0, 0]
    mu = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=300,
                                       monotonic_cst=mono, random_state=0).fit(X, y)
    # cross-fit residuals so that sigma is not understated
    oof = np.zeros(len(train))
    g = train.match_id.values
    for tr, te in GroupKFold(4).split(X, y, g):
        mm = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=300,
                                           monotonic_cst=mono, random_state=0).fit(X[tr], y[tr])
        oof[te] = mm.predict(X[te])
    sg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_leaf_nodes=12, min_samples_leaf=400,
                                       random_state=0).fit(X, (y - oof) ** 2)
    return mu, sg


def make_grid(mu, sg):
    """tabulate mu and sigma on (ci, yi, balls_left 0..120, wkts 0..10)."""
    ci, yi, bl, wk = np.meshgrid(np.arange(2), np.arange(5), np.arange(121), np.arange(11), indexing="ij")
    X = np.c_[bl.ravel(), np.minimum(wk.ravel(), 9), ci.ravel(), yi.ravel()]
    MU = mu.predict(X).reshape(ci.shape)
    SG = np.sqrt(np.clip(sg.predict(X), 4, None)).reshape(ci.shape)
    MU = np.clip(MU, 0, None)
    MU[:, :, 0, :] = 0; MU[:, :, :, 10] = 0
    SG = np.maximum(SG, 1.5)
    # enforce monotone-in-balls and wickets smoothing for sigma (cummax along balls)
    return MU, SG


def wp_raw(kind, ci, yi, score, bl, wk, req=None, par=None, MU=None, SG=None, sd2=30.0):
    """raw (uncalibrated) win prob of the BATTING side from a given pre-delivery state; all arrays."""
    bl = np.clip(bl, 0, 120).astype(int); wk = np.clip(wk, 0, 10).astype(int)
    mu = MU[ci, yi, bl, wk]; sg = SG[ci, yi, bl, wk]
    if kind == "chase":
        p = norm.cdf((mu - req + 0.5) / sg)
        p = np.where(req <= 0, 1.0, np.where((bl == 0) | (wk >= 10), 0.0, p))
    else:
        proj = score + mu
        p = norm.cdf((proj - par) / np.sqrt(sg ** 2 + sd2 ** 2))
    return p


def cal(p, ab):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return expit(ab[0] + ab[1] * logit(p))


def fit_platt(p, y):
    lr = LogisticRegression(C=1e6).fit(logit(np.clip(p, 1e-4, 1 - 1e-4)).reshape(-1, 1), y)
    return (float(lr.intercept_[0]), float(lr.coef_[0, 0]))


def par_table(train_matches_mask):
    t = b[train_matches_mask & (b.innings == 1)].groupby(["match_id"]).agg(tot=("score_after", "max"), ci=("ci", "first"), yi=("yi", "first"))
    par = t.groupby(["ci", "yi"]).tot.mean()
    sd = t.tot.std()
    return par, sd


# ------------------------------------------------------------------ match-grouped CV of the win-probability models
lab = b[b.bat_wins_match.notna()].copy()
cv_rows = []
oof_p = {"chase": [], "inn1": []}
matches = b.match_id.unique()
folds = np.array_split(rng.permutation(matches), 5)
for k, te_m in enumerate(folds):
    tr_mask = ~b.match_id.isin(te_m)
    mu, sg = fit_mu_sigma(b[tr_mask & (b.innings == 1)])
    MU, SG = make_grid(mu, sg)
    par, sd1 = par_table(tr_mask)
    for kind, inn in [("chase", 2), ("inn1", 1)]:
        d = lab[(lab.innings == inn)]
        dtr, dte = d[~d.match_id.isin(te_m)], d[d.match_id.isin(te_m)]
        def raw(df):
            if kind == "chase":
                return wp_raw("chase", df.ci.values, df.yi.values, df.score_before.values, df.balls_left.values, df.wkts_before.values, req=df.runs_req.values, MU=MU, SG=SG)
            pr = np.array([par.get((c_, y_), par.mean()) for c_, y_ in zip(df.ci, df.yi)])
            return wp_raw("inn1", df.ci.values, df.yi.values, df.score_before.values, df.balls_left.values, df.wkts_before.values, par=pr, MU=MU, SG=SG, sd2=sd1)
        ptr, pte = raw(dtr), raw(dte)
        ab = fit_platt(ptr, dtr.bat_wins_match.values)
        out = dte[["match_id", "balls_left", "wkts_before", "bat_wins_match"]].copy()
        out["p_raw"] = pte; out["p_cal"] = cal(pte, ab); out["kind"] = kind; out["fold"] = k
        if kind == "chase":
            rrr = np.where(dte.balls_left > 0, 6 * dte.runs_req / dte.balls_left.clip(lower=1), np.nan)
            out["rrr_only"] = expit(-(np.clip(rrr, 0, 40) - 8.0) * 0.55)   # naive RRR-only benchmark
            out["rrr_only"] = np.where(dte.runs_req <= 0, 1, out.rrr_only)
        oof_p[kind].append(out)
oof = {k: pd.concat(v) for k, v in oof_p.items()}
for k, d in oof.items():
    y = d.bat_wins_match.values
    r = {"n_rows": int(len(d)), "n_matches": int(d.match_id.nunique()),
         "brier_cal": float(brier_score_loss(y, d.p_cal)), "brier_raw": float(brier_score_loss(y, d.p_raw)),
         "brier_const": float(brier_score_loss(y, np.full(len(y), y.mean()))),
         "logloss_cal": float(log_loss(y, d.p_cal)), "logloss_const": float(log_loss(y, np.full(len(y), y.mean()))),
         "auc_cal": float(roc_auc_score(y, d.p_cal))}
    if k == "chase":
        r["brier_rrr_only"] = float(brier_score_loss(y, d.rrr_only.clip(1e-4, 1 - 1e-4)))
        r["auc_rrr_only"] = float(roc_auc_score(y, d.rrr_only))
        r["logloss_rrr_only"] = float(log_loss(y, d.rrr_only.clip(1e-4, 1 - 1e-4)))
    diag[f"wp_cv_{k}"] = r
    print(k, json.dumps(r, indent=1))
oof["chase"].to_pickle("oof_chase.pkl"); oof["inn1"].to_pickle("oof_inn1.pkl")

# ------------------------------------------------------------------ production fit on all data
mu, sg = fit_mu_sigma(b[b.innings == 1])
MU, SG = make_grid(mu, sg)
all_mask = np.ones(len(b), bool)
par, sd1 = par_table(all_mask)
diag["par_first_innings"] = {f"{'IPL' if c else 'T20I'}_{2022+y}": float(v) for (c, y), v in par.items()}
diag["sd_first_innings_total"] = float(sd1)
PAR = np.array([[par.get((c_, y_), par.mean()) for y_ in range(5)] for c_ in range(2)])
b["par"] = PAR[b.ci.values, b.yi.values]
lab = b[b.bat_wins_match.notna()].copy()

ab_chase = fit_platt(wp_raw("chase", *[lab[lab.innings == 2][c].values for c in ["ci", "yi", "score_before", "balls_left", "wkts_before"]], req=lab[lab.innings == 2].runs_req.values, MU=MU, SG=SG), lab[lab.innings == 2].bat_wins_match.values)
d1 = lab[lab.innings == 1]
ab_inn1 = fit_platt(wp_raw("inn1", d1.ci.values, d1.yi.values, d1.score_before.values, d1.balls_left.values, d1.wkts_before.values, par=d1.par.values, MU=MU, SG=SG, sd2=sd1), d1.bat_wins_match.values)
diag["platt_chase"] = ab_chase; diag["platt_inn1"] = ab_inn1


def wp_state(inn, ci, yi, score, bl, wk, req, par_):
    """calibrated win prob of the batting side for arbitrary (vectorised) states, terminal states exact."""
    if inn == 2:
        raw = wp_raw("chase", ci, yi, score, bl, wk, req=req, MU=MU, SG=SG)
        p = cal(raw, ab_chase)
        p = np.where(req <= 0, 1.0, np.where((bl <= 0) | (wk >= 10), 0.0, p))
    else:
        raw = wp_raw("inn1", ci, yi, score, bl, wk, par=par_, MU=MU, SG=SG, sd2=sd1)
        p = cal(raw, ab_inn1)
    return p


# ------------------------------------------------------------------ next-delivery outcome distribution
leg = b[b.legal].copy()
leg["o"] = np.select([leg.is_wicket, leg.runs == 0, leg.runs == 1, leg.runs == 2, leg.runs == 3, leg.runs.isin([4, 5]), leg.runs == 6],
                     [6, 0, 1, 2, 3, 4, 5], default=0)     # 0:0 1:1 2:2 3:3 4:4 5:6 6:W
leg["wg"] = np.minimum(leg.wkts_before // 3, 2)
RUNS_O = np.array([0, 1, 2, 3, 4, 6, 0])
cnt = leg.groupby(["innings", "over", "wg", "o"]).size().unstack("o").reindex(columns=range(7)).fillna(0) + 0.5
Pout = cnt.div(cnt.sum(1), axis=0)
b["wg"] = np.minimum(b.wkts_before // 3, 2)
key = pd.MultiIndex.from_arrays([b.innings, b.over, b.wg])
P = Pout.reindex(key).values                       # n x 7

# ------------------------------------------------------------------ win prob now and after each possible outcome
ci, yi = b.ci.values, b.yi.values
score, bl, wk = b.score_before.values, b.balls_left.values, b.wkts_before.values
req = b.runs_req.values; parv = b.par.values
inn = b.innings.values
p_now = np.where(inn == 2, wp_state(2, ci, yi, score, bl, wk, req, parv), wp_state(1, ci, yi, score, bl, wk, req, parv))
lev = np.zeros(len(b))
for o in range(7):
    r_o = RUNS_O[o]; w_o = 1 if o == 6 else 0
    p_after = np.where(inn == 2,
                       wp_state(2, ci, yi, score + r_o, bl - 1, wk + w_o, req - r_o, parv),
                       wp_state(1, ci, yi, score + r_o, bl - 1, wk + w_o, req, parv))
    lev += P[:, o] * np.abs(p_after - p_now)
b["p_bat"] = p_now
b["lev_raw"] = lev

# ------------------------------------------------------------------ Pressure Index
# (a) absolute leverage index: LI = E|dWP| / mean(E|dWP|); saturating map keeps the absolute scale (no rank flattening)
b["LI"] = b.lev_raw / b.lev_raw.mean()
b["L"] = 1 - np.exp(-b.LI / 2)
# (b) 'behind but still alive' stress (batter) / 'defending but opposition ahead & alive' stress (bowler)
b["D_bat"] = (1 - b.p_bat) * np.minimum(1, 4 * b.p_bat)
b["D_bowl"] = b.p_bat * np.minimum(1, 4 * (1 - b.p_bat))
# (c) collapse index: empirical CDF of wickets lost at this stage of innings 1 (uniform on [0,1])
ref = b[(b.innings == 1) & b.legal]
ecdf = {ov: np.sort(d.wkts_before.values) for ov, d in ref.groupby("over")}
def ecdf_at(ov, w):
    v = ecdf[ov]; lo = np.searchsorted(v, w, side="left"); hi = np.searchsorted(v, w, side="right")
    return (0.5 * (lo + hi)) / len(v)
b["collapse"] = [ecdf_at(o_, w_) for o_, w_ in zip(b.over.values, b.wkts_before.values)]
# (d) innings-1 end-of-innings intensity: data-driven scoring-intensity premium by over (0 = quietest over, 1 = death peak)
rpo = b.groupby("over").team_runs.mean()
PHI = ((rpo - rpo.min()) / (rpo.max() - rpo.min()))
b["phi"] = b.over.map(PHI)
diag["phi_by_over"] = {int(k): float(v) for k, v in PHI.items()}
b["C_bat"] = b.collapse; b["C_bowl"] = 1 - b.collapse


def build_pi(side, alpha=ALPHA, lam=LAMBDA_STAKE, w1=(0.40, 0.35, 0.25)):
    inn2 = alpha * b.L + (1 - alpha) * b[f"D_{side}"]
    inn1 = w1[0] * b.phi + w1[1] * b[f"D_{side}"] + w1[2] * b[f"C_{side}"]
    raw = np.where(b.innings == 2, inn2, inn1) * (1 + lam * b.stake)
    return pd.Series(raw, index=b.index)

for side in ["bat", "bowl"]:
    raw = build_pi(side)
    b[f"pi_raw_{side}"] = raw
    b[f"PI_{side}"] = raw.groupby(b.innings).rank(pct=True)
b["phase"] = pd.cut(b.over, [0, 6, 15, 20], labels=["Powerplay", "Middle", "Death"])

# alternate PI variants (used only in sensitivity analysis)
for lam in [0.0, 0.1, 0.4]:
    for side in ["bat", "bowl"]:
        b[f"PI_{side}_lam{int(lam*10)}"] = build_pi(side, lam=lam).groupby(b.innings).rank(pct=True)
for al in [0.4, 0.8]:
    for side in ["bat", "bowl"]:
        b[f"PI_{side}_al{int(al*10)}"] = build_pi(side, alpha=al).groupby(b.innings).rank(pct=True)
for side in ["bat", "bowl"]:
    b[f"PI_{side}_pool"] = build_pi(side).rank(pct=True)
for side in ["bat", "bowl"]:
    b[f"PI_{side}_ph"] = b.groupby("phase", observed=True)[f"pi_raw_{side}"].rank(pct=True)   # phase-neutral variant

# mu/lambda for wicket cost (runs lost if a wicket falls on this ball)
bl1 = np.clip(bl - 1, 0, 120)
MUg = MU[ci, yi]
b["lam_w"] = np.clip(MUg[np.arange(len(b)), bl1, np.minimum(wk, 10)] - MUg[np.arange(len(b)), bl1, np.minimum(wk + 1, 10)], 0.5, None)
b["mu_state"] = MUg[np.arange(len(b)), np.clip(bl, 0, 120), np.minimum(wk, 10)]

np.save("MU.npy", MU); np.save("SG.npy", SG)
b.drop(columns=["swap"], errors="ignore").to_pickle("balls_pi.pkl")
json.dump(diag, open("pressure_diag.json", "w"), indent=2)
print("PI summary by innings/phase (mean PI_bat):")
print(b.groupby(["innings", "phase"], observed=True)[["PI_bat", "PI_bowl", "lev_raw", "p_bat"]].mean().round(3))
print("share of balls PI_bat>=0.75 by innings x phase:")
print((b.PI_bat >= 0.75).groupby([b.innings, b.phase], observed=True).mean().round(3))
