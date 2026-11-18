import pandas as pd, numpy as np, pickle, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
os.makedirs("fig", exist_ok=True)
NAVY, TEAL, ORANGE, GOLD, GREY, RED = "#1F3A5F", "#2A9D8F", "#E76F51", "#E9C46A", "#8D99AE", "#C8553D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444", "axes.labelcolor": "#222", "axes.titleweight": "bold", "axes.titlesize": 10,
                     "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})
b = pd.read_pickle("balls_pi.pkl"); bt = pd.read_pickle("bat_rows.pkl")
m = pd.read_pickle("matches_clean.pkl").set_index("match_id")
fin = pickle.load(open("fin_ranked.pkl", "rb")); final = pickle.load(open("final_results.pkl", "rb"))
sens = pickle.load(open("sens_results.pkl", "rb"))
names = pd.read_pickle("names.pkl").set_index("pid").label

# ------------------------------------------------------------ F1: where does the PI put 'high pressure'?
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
hi = (b.PI_bat >= 0.75)
sh = hi.groupby([b.innings, b.over]).mean().unstack(0)
ax[0].plot(sh.index, sh[1], color=TEAL, lw=2, marker="o", ms=3, label="Innings 1 (setting)")
ax[0].plot(sh.index, sh[2], color=ORANGE, lw=2, marker="o", ms=3, label="Innings 2 (chasing)")
ax[0].axhline(0.25, color=GREY, ls=":", lw=1); ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax[0].set_xlabel("Over"); ax[0].set_ylabel("Balls flagged high-pressure"); ax[0].set_title("a) Share of balls in the top PI quartile"); ax[0].legend(frameon=False, fontsize=8)
c = b[(b.innings == 2) & (b.over >= 16)]
bins = np.linspace(0, 1, 11); cut = pd.cut(c.p_bat, bins, include_lowest=True)
v = (c.PI_bat >= 0.75).groupby(cut, observed=True).mean(); w = (c.PI_bowl >= 0.75).groupby(cut, observed=True).mean()
x = np.arange(len(v)); ax[1].bar(x - 0.2, v.values, 0.4, color=ORANGE, label="Batting side PI"); ax[1].bar(x + 0.2, w.values, 0.4, color=NAVY, label="Bowling side PI")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{int(a*100)}" for a in bins[:-1]], fontsize=8); ax[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax[1].set_xlabel("Batting side win probability (%, bucket lower bound)"); ax[1].set_title("b) Chase overs 16-20: PI peaks in live games"); ax[1].set_ylim(0, 1.25); ax[1].set_yticks([0, .25, .5, .75, 1]); ax[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
fig.tight_layout(); fig.savefig("fig/f1_pi_shape.png"); plt.close()

# ------------------------------------------------------------ F2: anatomy of a pressure chase (Asia Cup 2025 final)
cc_ = b[(b.innings == 2) & (b.stake == 0)].groupby("match_id").agg(s=("lev_raw", "sum"), n=("legal", "sum"), sw=("p_bat", lambda x: x.max() - x.min()))
cc_ = cc_[(cc_.n >= 110) & (cc_.sw > 0.6)]
mid = cc_.s.idxmax()
print("worked example match", mid)
d = b[(b.match_id == mid) & (b.innings == 2)].reset_index(drop=True)
mi = m.loc[mid]
fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
x = np.arange(len(d)); ax[0].plot(x, d.p_bat, color=NAVY, lw=1.8); ax[0].set_ylim(0, 1); ax[0].set_ylabel("Chasing side\nwin probability")
wk = d.index[d.is_wicket]; wk = wk[wk + 1 < len(d)] + 1; ax[0].scatter(wk, d.p_bat[wk], color=RED, s=22, zorder=3, label="Wicket"); ax[0].legend(frameon=False, loc="lower left", fontsize=8)
ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax[1].fill_between(x, d.PI_bat, color=ORANGE, alpha=0.35, step="mid"); ax[1].plot(x, d.PI_bat, color=ORANGE, lw=1.2, label="PI (batting side)")
ax[1].plot(x, d.PI_bowl, color=TEAL, lw=1.2, label="PI (bowling side)"); ax[1].axhline(0.75, color="#444", ls=":", lw=1); ax[1].text(len(d) - 1, 0.77, "high-pressure threshold", ha="right", fontsize=7)
ax[1].set_ylabel("Pressure Index"); ax[1].set_xlabel("Delivery number in the chase"); ax[1].set_ylim(0, 1.02); ax[1].legend(frameon=False, ncol=2, fontsize=8, loc="lower left")
ax[0].set_title(f"Worked example: {mi.home_team} v {mi.away_team}, {mi.match_date} ({mi.stage}) - chase of {int(d.target.iloc[0])}")
fig.tight_layout(); fig.savefig("fig/f2_chase_anatomy.png"); plt.close()

# ------------------------------------------------------------ F3: dose-response
f = bt.copy(); f["dec"] = pd.cut(f.PI_bat, np.linspace(0, 1, 11), include_lowest=True, labels=False)
g = f.groupby("dec").agg(rpb=("r_bat", "mean"), dot=("dot_bat", "mean"), bd=("bd", "mean"), out=("out_bat", "mean"), n=("r_bat", "size"))
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.3))
for a, (col, ttl, scale, fmt) in zip(ax, [("rpb", "Runs per over", 6, None), ("dot", "Dot-ball %", 100, "%"), ("bd", "Boundary %", 100, "%"), ("out", "Dismissal %", 100, "%")]):
    a.bar(g.index, g[col] * scale, color=[GREY] * 5 + [ORANGE] * 3 + [RED] * 2, width=0.8)
    a.set_title(ttl, fontsize=9); a.set_xticks([0, 4, 9]); a.set_xticklabels(["low", "mid", "high"], fontsize=8)
    lo, hi_ = (g[col] * scale).min(), (g[col] * scale).max(); a.set_ylim(lo - 0.4 * (hi_ - lo), hi_ + 0.1 * (hi_ - lo))
ax[0].set_xlabel("PI decile"); fig.suptitle("League-wide batter outcomes by Pressure Index decile (raw, unadjusted)", y=1.04, fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig("fig/f3_dose_response.png"); plt.close()

# ------------------------------------------------------------ F4: calibration of win-probability models
oc, oi = pd.read_pickle("oof_chase.pkl"), pd.read_pickle("oof_inn1.pkl"); dg = json.load(open("pressure_diag.json"))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for a, d_, ttl, key in [(ax[0], oc, "Chase (innings 2)", "wp_cv_chase"), (ax[1], oi, "Innings 1 (batting first)", "wp_cv_inn1")]:
    q = pd.qcut(d_.p_cal, 12, duplicates="drop"); r = d_.groupby(q, observed=True).agg(p=("p_cal", "mean"), y=("bat_wins_match", "mean"), n=("p_cal", "size"))
    a.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1); a.plot(r.p, r.y, color=NAVY, marker="o", ms=4, lw=1.6)
    a.set_title(ttl); a.set_xlabel("Predicted win probability (out-of-fold)"); a.set_xlim(0, 1); a.set_ylim(0, 1)
    k = dg[key]; txt = f"Brier {k['brier_cal']:.3f} (constant {k['brier_const']:.3f})\nAUC {k['auc_cal']:.2f}"
    if "brier_rrr_only" in k: txt += f"\nRRR-only Brier {k['brier_rrr_only']:.3f}"
    a.text(0.97, 0.05, txt, transform=a.transAxes, ha="right", va="bottom", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
ax[0].set_ylabel("Observed win frequency"); fig.tight_layout(); fig.savefig("fig/f4_calibration.png"); plt.close()

# ------------------------------------------------------------ F5: out-of-sample predictive validation
pv = pickle.load(open("pred_validation.pkl", "rb")); bv = pickle.load(open("blend_validation.pkl", "rb"))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
lab = ["Raw pressure\nvalue", "Shrunk\n(pressure only)", "50/50 blend\n(final)", "Hierarchical\n(borrows normal)"]
for a, side, ttl in [(ax[0], "bat", "Batters"), (ax[1], "bowl", "Bowlers")]:
    cols = [pv[side].val_raw_pressure, pv[side].val_A_shrunk_pressure_only, bv[side]["val_0.5"], pv[side].val_B_hierarchical]
    mu = [c.mean() for c in cols]; lo = [np.percentile(c, 10) for c in cols]; hi_ = [np.percentile(c, 90) for c in cols]
    a.bar(range(4), mu, color=[GREY, TEAL, ORANGE, NAVY], width=0.65)
    a.errorbar(range(4), mu, yerr=[np.array(mu) - lo, np.array(hi_) - mu], fmt="none", ecolor="#333", capsize=3, lw=1)
    for i, v_ in enumerate(mu): a.text(i, v_ + 0.012, f"{v_:.2f}", ha="center", fontsize=8)
    a.set_xticks(range(4)); a.set_xticklabels(lab, fontsize=7.5); a.set_title(ttl)
ax[0].set_ylabel("Weighted corr. with held-out\npressure value (100 random splits)"); fig.tight_layout(); fig.savefig("fig/f5_predictive.png"); plt.close()

# ------------------------------------------------------------ F6: shrinkage
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
for a, side, ttl in [(ax[0], "bat", "Batters"), (ax[1], "bowl", "Bowlers")]:
    r = fin[side]; sc = a.scatter(r.Value_raw, r.Value_shr, c=np.log10(r.nP), s=14 + r.nP / 12, cmap="viridis_r", alpha=0.85, edgecolor="white", lw=0.4)
    lim = [min(r.Value_raw.min(), r.Value_shr.min()) - 3, max(r.Value_raw.max(), r.Value_shr.max()) + 3]
    a.plot(lim, lim, color=GREY, ls="--", lw=1); a.axhline(0, color="#ddd", lw=0.8); a.axvline(0, color="#ddd", lw=0.8)
    for (i, row), off in zip(r[r.final_rank <= 3].iterrows(), [(-8, 7), (4, -10), (6, 5)]): a.annotate(row["name"].split()[-1], (row.Value_raw, row.Value_shr), fontsize=7, xytext=off, textcoords="offset points")
    a.set_xlabel("Raw pressure value (runs per 100 balls)", fontsize=8); a.set_title(ttl)
ax[0].set_ylabel("Shrunk (final) pressure value"); cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02); cb.set_label("log10 pressure balls", fontsize=8)
fig.savefig("fig/f6_shrinkage.png"); plt.close()

# ------------------------------------------------------------ F7: rank stability across the specification grid
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.9))
for a, side, ttl in [(ax[0], "bat", "Batters"), (ax[1], "bowl", "Bowlers")]:
    R = sens[side]["allranks"]; grid = [c for c in R.columns if c not in ("IPL rows only", "T20I rows only")]
    top = fin[side].head(14)
    for yi, (pid, row) in enumerate(top.iterrows()):
        rr = R.loc[pid, grid].values
        a.scatter(rr, np.full(len(rr), yi) + np.random.default_rng(yi).uniform(-0.18, 0.18, len(rr)), s=11, color=NAVY if row.final_rank <= 10 else GREY, alpha=0.5, lw=0)
        a.plot([np.median(rr)] * 2, [yi - 0.3, yi + 0.3], color=ORANGE, lw=2)
    a.axvspan(0, 10.5, color=TEAL, alpha=0.08); a.set_yticks(range(len(top))); a.set_yticklabels([f"{int(r_.final_rank)}. {r_['name']}" for _, r_ in top.iterrows()], fontsize=7.5)
    a.invert_yaxis(); a.set_xscale("log"); a.set_xticks([1, 2, 5, 10, 20, 40]); a.set_xticklabels(["1", "2", "5", "10", "20", "40"]); a.set_xlabel("Rank under each of 18 specifications"); a.set_title(ttl)
    a.axhline(9.5, color="#444", ls=":", lw=1)
fig.tight_layout(); fig.savefig("fig/f7_rank_stability.png"); plt.close()

# ------------------------------------------------------------ F8: component contributions
W = {"R": 0.25, "Wsave": 0.15, "Delta": 0.20, "DOT": 0.15, "BD": 0.10, "CONS": 0.15}
lab = {"R": "Scoring / run prevention", "Wsave": "Dismissal avoidance / wickets", "Delta": "Clutch differential", "DOT": "Dot-ball control", "BD": "Boundary hit / prevented", "CONS": "Consistency"}
colk = {"R": NAVY, "Wsave": TEAL, "Delta": GOLD, "DOT": ORANGE, "BD": "#9B5DE5", "CONS": GREY}
eff = {}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6))
for a, side, ttl in [(ax[0], "bat", "Batters"), (ax[1], "bowl", "Bowlers")]:
    r = final[side]["res"]; sdA, sdB = r.score_A.std(), r.score_B.std()
    contrib = pd.DataFrame({k: W[k] * (0.5 * r[f"{k}_A"] / sdA + 0.5 * r[f"{k}_B"] / sdB) for k in W}); contrib = contrib - contrib.mean()
    tot = contrib.sum(axis=1); eff[side] = {k: float(np.cov(contrib[k], tot)[0, 1] / tot.var()) for k in W}
    top = fin[side].head(10).index; cc = contrib.loc[top]; y = np.arange(len(top)); left_p = np.zeros(len(top)); left_n = np.zeros(len(top))
    for k in W:
        v_ = cc[k].values; pos = np.where(v_ > 0, v_, 0); neg = np.where(v_ < 0, v_, 0)
        a.barh(y, pos, left=left_p, color=colk[k], label=lab[k], height=0.7); a.barh(y, neg, left=left_n, color=colk[k], height=0.7); left_p += pos; left_n += neg
    a.set_yticks(y); a.set_yticklabels([names[i] for i in top], fontsize=7.5); a.invert_yaxis(); a.axvline(0, color="#333", lw=0.8)
    a.set_xlabel("Contribution to composite score", fontsize=8.5); a.set_title(ttl)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); fig.savefig("fig/f8_components.png"); plt.close()
json.dump(eff, open("effective_weights.json", "w"), indent=1)
print("effective weight share:", json.dumps({s: {k: round(v, 2) for k, v in d.items()} for s, d in eff.items()}))
print("figs done:", sorted(os.listdir("fig")))
