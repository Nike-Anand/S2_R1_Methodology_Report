"""Export the final ranked lists as CSV (full ranking with all components, plus the Top-10 submission file)."""
import os, pickle
import pandas as pd

OUT = os.environ.get("OUT_DIR", "outputs"); os.makedirs(OUT, exist_ok=True)
fin = pickle.load(open("fin_ranked.pkl", "rb"))
common = ["final_rank", "name", "consensus_mean_rank", "top10_share", "best_rank", "worst_rank", "base_rank", "score", "nP",
          "P_matches", "Value_raw", "Value_shr", "R_shr", "Wsave_shr", "Delta_shr", "DOT_shr", "BD_shr", "CONS_shr",
          "p_top10", "p_top20", "boot_rank_lo", "boot_rank_hi", "P_ipl_share"]
cols = {"bat": common + ["P_balls", "P_SR", "P_avg", "P_bpd", "P_dot%", "P_bdry%", "P_outs", "N_SR", "N_dot%", "N_bdry%"],
        "bowl": common + ["P_balls", "P_econ", "P_wkts", "P_bpw", "P_dot%", "P_bdry%", "N_econ", "N_dot%", "N_bdry%"]}
files = {"bat": "batters_full_ranking.csv", "bowl": "bowlers_full_ranking.csv"}
for side in ["bat", "bowl"]:
    d = fin[side][cols[side]].copy()
    d.insert(0, "player_id", d.index)
    d = d.rename(columns={"score": "base_blend_score", "top10_share": "share_of_18_specs_in_top10", "nP": "pressure_balls"})
    d.round(4).to_csv(os.path.join(OUT, files[side]), index=False)
top = [dict(category=lab, rank=int(x.final_rank), player=x["name"], player_id=pid)
       for side, lab in [("bat", "Batter"), ("bowl", "Bowler")] for pid, x in fin[side].head(10).iterrows()]
pd.DataFrame(top).to_csv(os.path.join(OUT, "final_top10_submission.csv"), index=False)
print("wrote", os.listdir(OUT))
