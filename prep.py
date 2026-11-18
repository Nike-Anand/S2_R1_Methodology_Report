"""
Step 1 - Data audit + state reconstruction.
Outputs: balls.pkl (one row per delivery with pre-delivery match state), matches_clean.pkl, audit.json
"""
import json, os
import numpy as np
import pandas as pd

U = os.environ.get("DATA_DIR", "data").rstrip("/") + "/"      # folder containing the supplied CSVs
b = pd.read_csv(U + "ball_by_ball.csv")
m = pd.read_csv(U + "matches.csv")
c = pd.read_csv(U + "competitions.csv")

audit = {}
audit["raw_balls"] = len(b)
audit["raw_matches_bbb"] = int(b.match_id.nunique())
audit["raw_matches_table"] = len(m)

# ---------------------------------------------------------------- match metadata
m = m.merge(c, on="comp_id", how="left")
m["ctype"] = np.where(m.comp_name.str.startswith("IPL"), "IPL",
                      np.where(m.match_format == "T20I", "T20I", "OTHER"))
m["year"] = pd.to_datetime(m.match_date).dt.year
res = m.result.str.lower()
m["dls"] = m.revised_target.notna() | m.result.str.lower().str.contains(r"d/l|dls|duckworth")
m["no_result"] = res.str.contains("no result|abandon")
m["tied"] = res.str.contains("tied")
m["has_bbb"] = m.match_id.isin(b.match_id)
# stage -> stakes class
stake_map = {"League": 0.0, "Others": 0.0, "Super 4": 0.5,
             "Qualifier": 1.0, "Eliminator": 1.0, "Semi Final": 1.0, "Final": 1.0}
m["stake"] = m.stage.map(stake_map).fillna(0.0)

excl = pd.DataFrame({
    "no_ballbyball": ~m.has_bbb,
    "non_IPL_non_T20I (Emerging Teams)": (m.ctype == "OTHER"),
    "DLS / revised target": m.dls,
    "no result": m.no_result,
})
audit["match_exclusions_overlapping"] = {k: int(v.sum()) for k, v in excl.items()}
keep = m.has_bbb & (m.ctype != "OTHER") & ~m.dls & ~m.no_result
audit["matches_kept"] = int(keep.sum())
m["keep"] = keep

# ---------------------------------------------------------------- ball level
b["et"] = b.extras_type.fillna("none")
b = b[b.et != "Penalty"].copy()                       # 1 row, 5 penalty runs, not a delivery
b = b[b.match_id.isin(m.loc[m.keep, "match_id"])]
audit["super_over_balls_dropped"] = int((b.innings > 2).sum())
b = b[b.innings <= 2].copy()
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

# legal delivery: wides, no-balls (type 'Overstep') and 'other' (no-ball variants) are illegal
b["is_wide"] = b.et == "wide"
b["is_nb"] = b.et.isin(["Overstep", "other"])
b["legal"] = ~(b.is_wide | b.is_nb)
b["is_bye"] = b.et == "byes"
b["is_lb"] = b.et == "legbyes"

grp = b.groupby(["match_id", "innings"], sort=False)
b["legal_before"] = grp.legal.cumsum() - b.legal.astype(int)
b["balls_left"] = 120 - b.legal_before
b["over"] = b.overid                                    # 1..20
# ---- strike rotation parity (to infer extras runs that the file does not record)
b["next_striker"] = grp.striker_id.shift(-1)
b["next_over"] = grp.overid.shift(-1)
mid = (b.next_over == b.overid) & ~b.is_wicket
b["swap"] = np.where(mid, b.next_striker != b.striker_id, np.nan)
b.drop(columns=["next_striker", "next_over"], inplace=True)

# extras reconstruction cells (counts), values calibrated below against known winning margins
b["cell_lb_odd"] = (b.is_lb & (b.swap == 1)).astype(int)
b["cell_lb_even"] = (b.is_lb & (b.swap == 0)).astype(int)
b["cell_lb_unk"] = (b.is_lb & b.swap.isna()).astype(int)
b["cell_bye_odd"] = (b.is_bye & (b.swap == 1)).astype(int)
b["cell_bye_even"] = (b.is_bye & (b.swap == 0)).astype(int)
b["cell_bye_unk"] = (b.is_bye & b.swap.isna()).astype(int)
b["cell_wd_run"] = (b.is_wide & (b.swap == 1)).astype(int)         # wide + a run completed
b["base_runs"] = (b.runs + b.extras_2_runs + b.is_wide.astype(int) + b.is_nb.astype(int))

cells = ["cell_lb_odd", "cell_lb_even", "cell_lb_unk", "cell_bye_odd", "cell_bye_even", "cell_bye_unk", "cell_wd_run"]
inn = b.groupby(["match_id", "innings"])[["base_runs"] + cells].sum().unstack("innings")
inn.columns = [f"{a}_{i}" for a, i in inn.columns]
inn = inn.dropna()
mm = m.set_index("match_id").join(inn, how="inner")
rm = mm.result.str.extract(r"[Ww]on [Bb]y (\d+) [Rr]uns?")[0].astype(float)
mm["margin"] = rm
# only matches won by the team batting first, so margin = total1 - total2
first_bat = b[(b.innings == 1)].groupby("match_id").batting_team_id.first()
mm["first_bat"] = first_bat
mm = mm[mm.margin.notna() & (mm.win_team_id == mm.first_bat)].copy()
audit["n_matches_margin_calibration"] = len(mm)

y = mm.margin - (mm.base_runs_1 - mm.base_runs_2)
X = pd.DataFrame({k: mm[f"{k}_1"] - mm[f"{k}_2"] for k in cells})
from sklearn.linear_model import Ridge
# non-negative least squares with a weak prior -> physically sensible values
from scipy.optimize import nnls
coef, _ = nnls(X.values, y.values)
vals = dict(zip(cells, coef))
audit["extras_run_values_fitted_NNLS"] = {k: round(float(v), 3) for k, v in vals.items()}
print("NNLS extras values:", audit["extras_run_values_fitted_NNLS"])

# validation: exact-match rate of reconstructed margin
pred0 = (mm.base_runs_1 - mm.base_runs_2)
pred1 = pred0 + X.values @ coef
pred_naive = pred0 + X.assign(
    cell_lb_odd=0, cell_lb_even=0, cell_lb_unk=0, cell_bye_odd=0, cell_bye_even=0, cell_bye_unk=0, cell_wd_run=0).values.sum(1)
audit["margin_check"] = {
    "n": int(len(mm)),
    "MAE_off_bat_plus_wd_nb_only": float((mm.margin - pred0).abs().mean()),
    "MAE_with_imputed_extras": float((mm.margin - pred1).abs().mean()),
    "exact_within_1_run_off_bat_only": float(((mm.margin - pred0).abs() <= 1).mean()),
    "exact_within_1_run_with_imputed": float(((mm.margin - pred1).abs() <= 1).mean()),
    "exact_match_with_imputed": float(((mm.margin - pred1).abs() < 0.5).mean()),
}
print(audit["margin_check"])

# apply values to every delivery
b["team_runs"] = b.base_runs + sum(b[k] * vals[k] for k in cells)
b["bowler_runs"] = b.runs + b.extras_2_runs * 0 + b.is_wide.astype(int) + b.is_nb.astype(int) + \
    b.cell_wd_run * vals["cell_wd_run"]                     # byes / leg-byes are NOT charged to the bowler

# ---------------------------------------------------------------- running state (pre-delivery)
b["team_wkt"] = b.is_wicket.astype(int)
grp = b.groupby(["match_id", "innings"], sort=False)
b["score_before"] = grp.team_runs.cumsum() - b.team_runs
b["wkts_before"] = grp.team_wkt.cumsum() - b.team_wkt
b["wkts_before"] = b.wkts_before.clip(upper=9)
b["score_after"] = b.score_before + b.team_runs

# innings-1 final total -> chase target
tot1 = b[b.innings == 1].groupby("match_id").score_after.max()
b["target"] = b.match_id.map(tot1) + 1
b["runs_req"] = np.where(b.innings == 2, b.target - b.score_before, np.nan)

# match result label (batting-first team wins?)
mm2 = m.set_index("match_id")
fb = b[b.innings == 1].groupby("match_id").batting_team_id.first()
b["bat_first_team"] = b.match_id.map(fb)
b["bf_win"] = b.match_id.map(mm2.win_team_id) == b.bat_first_team
b["bf_win"] = b.bf_win.astype(float)
b.loc[b.match_id.map(mm2.win_team_id).isna() | b.match_id.map(mm2.tied), "bf_win"] = np.nan
b["bat_wins_match"] = np.where(b.innings == 1, b.bf_win, 1 - b.bf_win)
b["ctype"] = b.match_id.map(mm2.ctype)
b["stage"] = b.match_id.map(mm2.stage)
b["stake"] = b.match_id.map(mm2.stake)
b["year"] = b.match_id.map(mm2.year)
b["date"] = pd.to_datetime(b.match_id.map(mm2.match_date))
b["comp_id"] = b.match_id.map(mm2.comp_id)
b["venue_id"] = b.match_id.map(mm2.venue_id)
b["season_key"] = b.ctype + "_" + b.year.astype(str)

# sanity: chase teams that "won" should reach target
last2 = b[b.innings == 2].groupby("match_id").agg(final=("score_after", "max"), tgt=("target", "first"),
                                                     wk=("wkts_before", "max"))
last2["win2"] = last2.index.map(lambda i: mm2.win_team_id.get(i)) == b[b.innings == 2].groupby("match_id").batting_team_id.first()
audit["chase_winners_reaching_target_pct"] = float((last2[last2.win2].final >= last2[last2.win2].tgt - 1).mean())
audit["chase_losers_below_target_pct"] = float((last2[~last2.win2].final < last2[~last2.win2].tgt).mean())


# ---- curtailed / anomalous innings that the revised-overs fields do not flag
ii = b.groupby(["match_id", "innings"]).agg(legal=("legal", "sum"), wk=("team_wkt", "sum"),
                                              final=("score_after", "max"), tgt=("target", "first")).unstack("innings")
ii.columns = [f"{a}{i}" for a, i in ii.columns]
bat2 = b[b.innings == 2].groupby("match_id").batting_team_id.first()
ii["win2"] = (ii.index.map(mm2.win_team_id) == bat2.reindex(ii.index))
bad1 = ii[(ii.legal1 < 120) & (ii.wk1 < 10)].index
bad2 = ii[(ii.legal2 < 120) & (ii.wk2 <= 8) & (~ii.win2)].index
bad3 = ii[ii.win2 & ((ii.final2 - ii.tgt2) < -8)].index
bad4 = ii[ii.legal2.isna()].index
anom = set(bad1) | set(bad2) | set(bad3) | set(bad4)
audit["anomalous_matches_dropped"] = {"inn1_curtailed": len(bad1), "inn2_curtailed_lost": len(bad2),
                                      "chase_winner_short_of_target": len(bad3), "missing_innings": len(bad4),
                                      "total_unique": len(anom)}
b = b[~b.match_id.isin(anom)].copy()
m.loc[m.match_id.isin(anom), "keep"] = False
audit["matches_kept"] = int(m.keep.sum())

# innings sanity
ii = b.groupby(["match_id", "innings"]).agg(legal=("legal", "sum"), wk=("team_wkt", "sum"))
audit["innings_legal_balls_gt120"] = int((ii.legal > 120).sum())
audit["innings_wkts_gt10"] = int((ii.wk > 10).sum())
audit["balls_kept"] = len(b)
audit["matches_kept_with_both_innings"] = int(b.groupby("match_id").innings.nunique().eq(2).sum())

# stable player table (ids are the key - 3 names map to 2 ids)
names = pd.concat([b[["striker_id", "striker_name"]].rename(columns={"striker_id": "pid", "striker_name": "name"}),
                   b[["bowler_id", "bowler_name"]].rename(columns={"bowler_id": "pid", "bowler_name": "name"})]).drop_duplicates()
dup = names.groupby("name").pid.nunique()
dup = dup[dup > 1].index
names["label"] = names.name
for nm in dup:
    ids = sorted(names.loc[names.name == nm, "pid"].unique())
    for k, i in enumerate(ids):
        names.loc[(names.name == nm) & (names.pid == i), "label"] = f"{nm} (id {i})"
names = names.drop_duplicates("pid")
audit["duplicate_names_disambiguated"] = list(dup)
names.to_pickle("names.pkl")
pname = names.set_index("pid").label
b["striker"] = b.striker_id.map(pname)
b["bowler"] = b.bowler_id.map(pname)

b.to_pickle("balls.pkl")
m.to_pickle("matches_clean.pkl")
json.dump(audit, open("audit.json", "w"), indent=2, default=str)
print(json.dumps(audit, indent=2, default=str))
