"""
Step 3 - Expected-performance baselines and per-ball value (runs above expectation, wicket-adjusted).
Context model  : HistGradientBoosting on (over, wickets lost, innings, competition type, season, free-hit)
Opposition     : ridge/BLUP effects for opposing player and opposing team, subtracted from each ball's residual
Value currency : V_bat  = (r - r_hat - opp) - lambda_w * (out - out_hat - opp)           [runs above expectation]
                 V_bowl = -(runs_conceded - c_hat - opp) + lambda_w * (wkt - wkt_hat - opp)
"""
import json, warnings
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
b = pd.read_pickle("balls_pi.pkl")
diag = {}

b["prev_nb"] = b.groupby(["match_id", "innings"]).is_nb.shift(1).fillna(False).astype(int)
b["ci"] = (b.ctype == "IPL").astype(int)
CTX = ["over", "wkts_before", "innings", "ci", "yi", "prev_nb"]

# --- outcome flags
bat_dis = b.is_wicket & (b.dismissed_player_id == b.striker_id) & (b.dismissal_type != "Retire Out")
bowl_dis = b.is_wicket & b.dismissal_type.isin(["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"])
b["r_bat"] = b.runs.astype(float)
b["dot_bat"] = (b.runs == 0).astype(float)
b["bd"] = (b.is_four | b.is_six).astype(float)
b["out_bat"] = bat_dis.astype(float)
b["wkt_bowl"] = bowl_dis.astype(float)
b["dot_bowl"] = (b.bowler_runs == 0).astype(float) * b.legal
b["runs_bowl"] = b.bowler_runs.astype(float)


def ctx_pred(df, target, kind):
    X = df[CTX].values
    kw = dict(max_iter=120, learning_rate=0.08, max_leaf_nodes=12, min_samples_leaf=400, random_state=0,
              categorical_features=[False] * len(CTX))
    mdl = (HistGradientBoostingRegressor(**kw) if kind == "reg" else HistGradientBoostingClassifier(**kw)).fit(X, df[target].values)
    return mdl.predict(X) if kind == "reg" else mdl.predict_proba(X)[:, 1]


def eb_alpha(res, ids, min_n=60):
    g = pd.DataFrame({"r": res, "i": ids}).groupby("i").r.agg(["mean", "count"])
    s2 = np.var(res)
    gg = g[g["count"] >= min_n]
    tau2 = max(np.average((gg["mean"] - np.average(gg["mean"], weights=gg["count"])) ** 2, weights=gg["count"]) - np.mean(s2 / gg["count"]), 1e-6)
    return s2 / tau2


def opp_effects(res, own, opp, opp_team, alpha, team_scale=3.0):
    """BLUP-style ridge: res_i = own_effect + opp_effect + opp_team_effect; return the opposition part per row."""
    io, uo = pd.factorize(own); ip, up = pd.factorize(opp); it, ut = pd.factorize(opp_team)
    n = len(res)
    r = np.arange(n)
    A = sp.hstack([sp.csr_matrix((np.ones(n), (r, io)), shape=(n, len(uo))),
                   sp.csr_matrix((np.ones(n), (r, ip)), shape=(n, len(up))),
                   sp.csr_matrix((np.full(n, team_scale), (r, it)), shape=(n, len(ut)))]).tocsr()
    sol = lsqr(A, res, damp=np.sqrt(alpha), atol=1e-7, btol=1e-7, iter_lim=400)[0]
    e_opp = sol[len(uo):len(uo) + len(up)]
    e_team = sol[len(uo) + len(up):] * team_scale
    e_own = sol[:len(uo)]
    opp_part = e_opp[ip] + e_team[it]
    return opp_part, pd.Series(e_own, index=uo), pd.Series(e_opp, index=up), pd.Series(e_team, index=ut)


# ================================================================== BATTER ROWS (balls faced: wides excluded)
bt = b[~b.is_wide].copy()
for t, k in [("r_bat", "reg"), ("dot_bat", "clf"), ("bd", "clf"), ("out_bat", "clf")]:
    bt[f"{t}_hat"] = ctx_pred(bt, t, k)
    res = bt[t].values - bt[f"{t}_hat"].values
    alpha = eb_alpha(res, bt.striker_id.values)
    opp, own_e, opp_e, team_e = opp_effects(res, bt.striker_id.values, bt.bowler_id.values, bt.bowling_team_id.values, alpha)
    bt[f"{t}_adj"] = res - opp
    diag[f"bat_{t}_alpha"] = float(alpha)
    diag[f"bat_{t}_opp_sd"] = float(np.std(opp))
bt["V_bat"] = bt.r_bat_adj - bt.lam_w * bt.out_bat_adj
bt["dot_pp"] = -100 * bt.dot_bat_adj          # fewer dots than expected = positive
bt["bd_pp"] = 100 * bt.bd_adj
keepb = ["match_id", "innings", "over", "phase", "striker_id", "striker", "bowler_id", "stage", "stake", "ctype", "year", "date",
         "PI_bat", "V_bat", "dot_pp", "bd_pp", "r_bat_adj", "out_bat_adj", "r_bat", "out_bat", "dot_bat", "bd", "r_bat_hat", "out_bat_hat", "lam_w", "p_bat", "balls_left", "wkts_before", "batting_order"] + \
        [c for c in bt.columns if c.startswith("PI_bat_")]
bt[keepb].to_pickle("bat_rows.pkl")
print("batter rows", len(bt), "V_bat mean", bt.V_bat.mean(), " (should be ~0)")

# ================================================================== BOWLER ROWS (all deliveries)
bw = b.copy()
for t, k in [("runs_bowl", "reg"), ("dot_bowl", "clf"), ("bd", "clf"), ("wkt_bowl", "clf")]:
    bw[f"{t}_hat"] = ctx_pred(bw, t, k)
    res = bw[t].values - bw[f"{t}_hat"].values
    alpha = eb_alpha(res, bw.bowler_id.values)
    opp, own_e, opp_e, team_e = opp_effects(res, bw.bowler_id.values, bw.striker_id.values, bw.batting_team_id.values, alpha)
    bw[f"{t}_adj"] = res - opp
    diag[f"bowl_{t}_alpha"] = float(alpha)
    diag[f"bowl_{t}_opp_sd"] = float(np.std(opp))
bw["V_bowl"] = -bw.runs_bowl_adj + bw.lam_w * bw.wkt_bowl_adj
bw["dot_pp"] = 100 * bw.dot_bowl_adj           # more dots than expected = positive
bw["bd_pp"] = -100 * bw.bd_adj                 # fewer boundaries conceded = positive
keepw = ["match_id", "innings", "over", "phase", "bowler_id", "bowler", "striker_id", "stage", "stake", "ctype", "year", "date", "legal",
         "PI_bowl", "V_bowl", "dot_pp", "bd_pp", "runs_bowl_adj", "wkt_bowl_adj", "runs_bowl", "wkt_bowl", "dot_bowl", "bd", "runs_bowl_hat", "wkt_bowl_hat", "lam_w", "p_bat", "balls_left", "wkts_before"] + \
        [c for c in bw.columns if c.startswith("PI_bowl_")]
bw[keepw].to_pickle("bowl_rows.pkl")
print("bowler rows", len(bw), "V_bowl mean", bw.V_bowl.mean())
json.dump(diag, open("values_diag.json", "w"), indent=2)
print(json.dumps(diag, indent=1))
