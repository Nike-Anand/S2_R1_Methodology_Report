import pandas as pd, numpy as np, pickle, json, os
from scipy.stats import spearmanr
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
                                KeepTogether, NextPageTemplate)
from reportlab.lib.utils import ImageReader
import matplotlib
FD = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/")
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf")); pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf")); pdfmetrics.registerFont(TTFont("DV-BI", FD + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY, TEAL, ORANGE, LIGHT, GREYT = colors.HexColor("#1F3A5F"), colors.HexColor("#2A9D8F"), colors.HexColor("#E76F51"), colors.HexColor("#F1F5F9"), colors.HexColor("#5B6B7C")
S = dict(
    title=ParagraphStyle("t", fontName="DV-B", fontSize=20, leading=25, textColor=NAVY, spaceAfter=4),
    sub=ParagraphStyle("s", fontName="DV", fontSize=10.5, leading=14, textColor=GREYT, spaceAfter=10),
    h1=ParagraphStyle("h1", fontName="DV-B", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=5),
    h2=ParagraphStyle("h2", fontName="DV-B", fontSize=10.3, leading=13.5, textColor=TEAL, spaceBefore=8, spaceAfter=3),
    body=ParagraphStyle("b", fontName="DV", fontSize=8.7, leading=12.3, spaceAfter=5, textColor=colors.HexColor("#1b1b1b")),
    bul=ParagraphStyle("bu", fontName="DV", fontSize=8.7, leading=12.1, leftIndent=11, bulletIndent=2, spaceAfter=2.5),
    eq=ParagraphStyle("eq", fontName="DV", fontSize=8.8, leading=13, alignment=TA_CENTER, textColor=NAVY),
    cap=ParagraphStyle("c", fontName="DV-I", fontSize=7.6, leading=10, textColor=GREYT, spaceAfter=7),
    cell=ParagraphStyle("cell", fontName="DV", fontSize=7.4, leading=9.4),
    cellb=ParagraphStyle("cellb", fontName="DV-B", fontSize=7.4, leading=9.4),
    head=ParagraphStyle("head", fontName="DV-B", fontSize=7.3, leading=9.2, textColor=colors.white),
    call=ParagraphStyle("call", fontName="DV", fontSize=8.7, leading=12.3, textColor=colors.HexColor("#1b1b1b")),
)
W_PAGE = A4[0] - 3.4 * cm


def P(t, st="body"): return Paragraph(t, S[st])
def H1(t): return Paragraph(t, S["h1"])
def H2(t): return Paragraph(t, S["h2"])
def BL(items): return [Paragraph(i, S["bul"], bulletText="•") for i in items]
def EQ(t):
    tb = Table([[Paragraph(t, S["eq"])]], colWidths=[W_PAGE]); tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return [tb, Spacer(1, 5)]
def CALL(t, color=TEAL):
    tb = Table([[Paragraph(t, S["call"])]], colWidths=[W_PAGE]); tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF6F4")), ("LINEBEFORE", (0, 0), (0, -1), 3, color), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return [tb, Spacer(1, 6)]
def FIG(path, width_cm, cap):
    w, h = ImageReader(path).getSize(); wd = width_cm * cm
    return KeepTogether([Image(path, width=wd, height=wd * h / w), Paragraph(cap, S["cap"])])
def TBL(data, colw, header=True, zebra=True, fs=None, align_right_from=None, bold_first_col=False):
    rows = []
    for i, r in enumerate(data):
        rows.append([Paragraph(str(c), S["head"] if (header and i == 0) else (S["cellb"] if (bold_first_col and j == 0) else S["cell"])) for j, c in enumerate(r)])
    t = Table(rows, colWidths=[c * cm for c in colw], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6), ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
          ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DCE4"))]
    if header: st += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st)); return t


# ------------------------------------------------------------------ data
fin = pickle.load(open("fin_ranked.pkl", "rb")); final = pickle.load(open("final_results.pkl", "rb")); sens = pickle.load(open("sens_results.pkl", "rb"))
audit = json.load(open("audit.json")); dg = json.load(open("pressure_diag.json")); perm = json.load(open("perm_results.json")); eff = json.load(open("effective_weights.json"))
pv = pickle.load(open("pred_validation.pkl", "rb")); bv = pickle.load(open("blend_validation.pkl", "rb"))
bt = pd.read_pickle("bat_rows.pkl"); bw = pd.read_pickle("bowl_rows.pkl")
names = pd.read_pickle("names.pkl").set_index("pid").label
fb, fw = fin["bat"], fin["bowl"]
Pb = bt[bt.PI_bat >= .75]; Nb = bt[bt.PI_bat <= .5]; Pw = bw[bw.PI_bowl >= .75]; Nw = bw[bw.PI_bowl <= .5]
lg = dict(sr_p=100 * Pb.r_bat.mean(), sr_n=100 * Nb.r_bat.mean(), out_p=100 * Pb.out_bat.mean(), out_n=100 * Nb.out_bat.mean(), dot_p=100 * Pb.dot_bat.mean(), dot_n=100 * Nb.dot_bat.mean(),
          bd_p=100 * Pb.bd.mean(), bd_n=100 * Nb.bd.mean(), ec_p=6 * Pw.runs_bowl.sum() / Pw.legal.sum(), ec_n=6 * Nw.runs_bowl.sum() / Nw.legal.sum(),
          dotw_p=100 * Pw.dot_bowl.sum() / Pw.legal.sum(), bdw_p=100 * Pw.bd.mean())
f = bt.copy(); f["dec"] = pd.cut(f.PI_bat, np.linspace(0, 1, 11), include_lowest=True, labels=False)
dec = f.groupby("dec").agg(rpo=("r_bat", lambda s: 6 * s.mean()), dot=("dot_bat", "mean"), bd=("bd", "mean"), out=("out_bat", "mean"))
vol_b = spearmanr(final["bat"]["res"].score, final["bat"]["res"].nP)[0]; vol_w = spearmanr(final["bowl"]["res"].score, final["bowl"]["res"].nP)[0]
tauR = final["bat"]["parts"]["A"]["R"]["tau"]; varR = final["bat"]["var"]["R"]
Bn = {n: tauR ** 2 / (tauR ** 2 + varR / n) for n in [20, 100, 300, 600]}
short = {"R": ("scoring", "run prevention"), "Wsave": ("dismissal avoidance", "wicket-taking"), "Delta": ("clutch gap", "clutch gap"), "DOT": ("dot-ball control", "dot-ball control"), "BD": ("boundary hitting", "boundary prevention"), "CONS": ("consistency", "consistency")}
WTS = {"R": 0.25, "Wsave": 0.15, "Delta": 0.20, "DOT": 0.15, "BD": 0.10, "CONS": 0.15}
def drivers(side):
    r = final[side]["res"]; sdA, sdB = r.score_A.std(), r.score_B.std()
    c = pd.DataFrame({k: WTS[k] * (0.5 * r[f"{k}_A"] / sdA + 0.5 * r[f"{k}_B"] / sdB) for k in WTS}); c = c - c.mean()
    idx = 0 if side == "bat" else 1
    return {pid: " + ".join(short[k][idx] for k in c.loc[pid].sort_values(ascending=False).index[:2]) for pid in c.index}
drv = {"bat": drivers("bat"), "bowl": drivers("bowl")}

# ------------------------------------------------------------------ document
OUT_DIR = os.environ.get("OUT_DIR", "outputs"); os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "SupeRR_S2_R1_Methodology_Report.pdf")
def footer(c, d):
    c.saveState(); c.setFont("DV", 7); c.setFillColor(GREYT)
    c.drawString(1.7 * cm, 1.0 * cm, "SupeRR Selector Season 2 · Round 1 · Pressure-adjusted rating of T20 batters and bowlers"); c.drawRightString(A4[0] - 1.7 * cm, 1.0 * cm, f"Page {d.page}")
    c.setStrokeColor(colors.HexColor("#D5DCE4")); c.line(1.7 * cm, 1.35 * cm, A4[0] - 1.7 * cm, 1.35 * cm); c.restoreState()
doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm, title="Methodology Report - SupeRR Selector S2 Round 1", author="Round 1 submission")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer)])
st = []

# ================================================================== 1. EXECUTIVE SUMMARY
st += [P("Who delivers when it matters?", "title"), P("A pressure-adjusted rating of T20 batters and bowlers (IPL and T20I, 2022-2026) · Methodology report, SupeRR Selector Season 2, Round 1", "sub")]
st += CALL(f"<b>What we did.</b> We scored all {audit['balls_kept']:,} deliveries from {audit['matches_kept']} matches with a <b>Pressure Index</b> (PI, 0-1) computed from the match state <i>before</i> each ball. Every ball was then compared with what an average player would be expected to do in that exact situation, adjusted for the opposition, with wickets converted into runs. Player ratings use <b>empirical-Bayes shrinkage</b> so small samples cannot beat proven performers, and the final order is a <b>consensus rank across 18 pre-defined model specifications</b> so that no single arbitrary design choice decides the submission.")
st.append(H1("1. Final submission"))
def top_table(side):
    r = (fb if side == "bat" else fw).head(10)
    if side == "bat":
        d = [["#", "Player", "Pressure balls (matches)", f"SR (league {lg['sr_p']:.0f})", "Avg", "Dot %", "Bdry %", "Value*", "Top-10 specs (of 18)", "Main drivers"]]
        for pid, x in r.iterrows():
            d.append([int(x.final_rank), f"<b>{x['name']}</b>", f"{int(x.P_balls)} ({int(x.P_matches)})", f"{x.P_SR:.0f}", "-" if np.isnan(x.P_avg) else f"{x.P_avg:.0f}", f"{x['P_dot%']:.0f}", f"{x['P_bdry%']:.0f}", f"{x.Value_shr:+.0f}", f"{int(round(x.top10_share*18))}", drv["bat"][pid]])
        cw = [0.6, 3.5, 1.9, 1.4, 0.9, 0.9, 0.9, 1.1, 1.4, 4.8]
    else:
        d = [["#", "Player", "Pressure balls (matches)", f"Econ (league {lg['ec_p']:.1f})", "Balls/ wkt", "Dot %", "Bdry %", "Value*", "Top-10 specs (of 18)", "Main drivers"]]
        for pid, x in r.iterrows():
            d.append([int(x.final_rank), f"<b>{x['name']}</b>", f"{int(x.P_balls)} ({int(x.P_matches)})", f"{x.P_econ:.1f}", "-" if np.isnan(x.P_bpw) else f"{x.P_bpw:.0f}", f"{x['P_dot%']:.0f}", f"{x['P_bdry%']:.0f}", f"{x.Value_shr*0.06:+.2f}", f"{int(round(x.top10_share*18))}", drv["bowl"][pid]])
        cw = [0.6, 3.5, 1.9, 1.5, 1.1, 0.9, 0.9, 1.1, 1.4, 4.5]
    return TBL(d, cw)
st += [KeepTogether([H2("Top 10 batters under pressure"), top_table("bat")]), Spacer(1, 3),
       P("<i>*Value = final (shrunk, opposition-adjusted) runs above expectation per 100 balls faced under pressure, with dismissals charged at their run cost. SR, average, dot and boundary % are the raw pressure-state figures for the player; they are not phase-adjusted, so use Value for comparisons.</i>", "cap")]
st += [KeepTogether([H2("Top 10 bowlers under pressure"), top_table("bowl")]), Spacer(1, 3),
       P("<i>*Value = final runs saved per over (including the run value of wickets) relative to expectation in the same situations against the same opposition. Pressure balls are legal deliveries; Econ and dot % are raw.</i>", "cap")]
st.append(H1("2. Headline findings"))
st += BL([
    f"<b>Pressure is measurable and behaves like pressure.</b> Our win-probability engine reaches AUC {dg['wp_cv_chase']['auc_cal']:.2f} and Brier {dg['wp_cv_chase']['brier_cal']:.3f} out-of-fold in chases (an RRR-only rule scores {dg['wp_cv_chase']['brier_rrr_only']:.3f}; a coin-flip {dg['wp_cv_chase']['brier_const']:.3f}). In overs 16-20 of chases, 99.7% of balls in genuinely close states (win probability 30-70%) are flagged high-pressure and none of the balls in already-decided states (win probability below 3% or above 97%) are. Across PI deciles, batter dismissal probability rises from {100*dec.out.iloc[:3].mean():.1f}% in the lowest three deciles to {100*dec.out.iloc[-1]:.1f}% in the top decile.",
    f"<b>Shrinkage matters a great deal.</b> In 100 random match splits, the final shrunk rating predicts a player's pressure performance in unseen matches about {bv['bat']['val_0.5'].mean()/pv['bat'].val_raw_pressure.mean():.1f}x (batters) and {bv['bowl']['val_0.5'].mean()/pv['bowl'].val_raw_pressure.mean():.1f}x (bowlers) better than raw pressure statistics.",
    f"<b>A separate 'clutch gene' is not detectable.</b> A permutation test finds no evidence that players differ in how their performance changes under pressure beyond chance (batters p = {perm['bat']['p_value']:.2f}, bowlers p = {perm['bowl']['p_value']:.3f}). What is detectable is persistent quality <i>in</i> pressure situations, which largely mirrors overall quality but is not identical to it. We therefore weight the differential modestly and let shrinkage set its effective weight (about {100*eff['bat']['Delta']:.0f}% for batters, {100*eff['bowl']['Delta']:.0f}% for bowlers).",
    "<b>For Rajasthan Royals (Section 11):</b> RR's pressure profile is strong at the top of the order (slots 1-2 rank 1st of 10 in IPL 2025-26) and in middle-overs spin, and weakest at the death (batting 8th, bowling 7th) and in slots 3-5 (7th). Pace bowlers who control the death (Bumrah, Pathirana, Ellis) and finishing or middle-order batters (Tim David, Naman Dhir, Suryakumar Yadav) offer the largest indicative gains; two of the 20 players, Russell and Nawaz, cannot be signed.",
    "<b>Confidence is highest at the top.</b> Abhishek Sharma and Sunil Narine rank first in most specifications (and never lower than third and second); the top 6 batters and top 5 bowlers each sit in the top 10 in at least 16 of 18 specifications. Batter places 7-10 are close calls (Section 8); we say so rather than hide it."])


# ================================================================== 3. DESIGN + DATA
st.append(H1("3. Design principles"))
st += BL(["<b>Pressure is defined before the ball is bowled</b> from match state only. Never from the final margin, because close finishes are partly caused by the players being rated (circular).",
          "<b>Pressure is perspective-specific.</b> A batter is stressed when his side is behind but still alive; a bowler is stressed when the batting side is ahead and still alive. Both are stressed when the game is on the line.",
          "<b>Per-ball residuals, never totals.</b> Volume cannot earn credit: a death-overs specialist is compared with the league-average player in the same over, wicket and competition state.",
          "<b>Estimate uncertainty explicitly.</b> Every rating is a shrunk estimate with bootstrap rank intervals.",
          "<b>Everything comes from the supplied files</b> and is reproducible end to end (Appendix B)."])
st.append(H1("4. Data audit and preparation"))
st.append(P(f"We used <font name='DV-B'>ball_by_ball</font> (200,999 deliveries, 867 matches), <font name='DV-B'>matches</font> (883), <font name='DV-B'>competitions</font> and <font name='DV-B'>teams</font>. The audit changed the build in several ways:"))
st += BL([
    "<b>Extras runs are not recorded.</b> <i>runs</i> is bat runs only and <i>extras1_runs</i> is a categorical type code, not a run count. Team totals, and therefore chase targets and required runs, had to be reconstructed. Wides and no-balls carry 1 extra run; leg-byes and byes are inferred from strike rotation on the next ball (odd or even run parity) with run values calibrated by non-negative least squares against 383 matches whose winning margin is known. Mean absolute error in the first-minus-second innings margin fell from "
    f"{audit['margin_check']['MAE_off_bat_plus_wd_nb_only']:.2f} to {audit['margin_check']['MAE_with_imputed_extras']:.2f} runs, with {100*audit['margin_check']['exact_within_1_run_with_imputed']:.0f}% of matches within 1 run (from {100*audit['margin_check']['exact_within_1_run_off_bat_only']:.0f}%). Residual error of 1-3 runs in required runs is small relative to the 15-25 run standard deviation of runs-to-go.",
    "<b>Legal deliveries.</b> Wides, no-balls (<i>Overstep</i>) and the <i>other</i> no-ball variants are not legal balls; with this rule 98.2% of overs contain exactly six legal balls. Free hits are flagged in the dismissal model.",
    "<b>Competitions.</b> The 141 competitions are 134 T20I series/tournaments, <b>5 IPL seasons (2022-26)</b> and 2 Emerging Teams Asia Cups (30 matches), which are neither IPL nor T20I; we excluded them from ratings and baselines. (The brief's '7 IPL tournaments' is therefore 5 IPL + 2 Emerging.)",
    "<b>Player identity.</b> Player IDs are the key. Two names (Abdul Samad, Haider Ali) map to two different IDs each and are disambiguated by ID.",
    "<b>Rain-affected and incomplete matches.</b> Matches with a revised target (or D/L text in the result), no result, no ball data, or curtailed innings not flagged by the revised-overs fields were removed so that required rates are meaningful. Super-over balls (26) were dropped."])
ex = audit["match_exclusions_overlapping"]
st += [Spacer(1, 3), TBL([["Stage", "Matches", "Notes"],
    ["Listed in matches.csv", "883", "867 have ball-by-ball data"],
    ["Excluded (overlapping reasons)", "105", f"no ball data {ex['no_ballbyball']}; Emerging Teams {ex['non_IPL_non_T20I (Emerging Teams)']}; revised target/D/L {ex['DLS / revised target']}; no result/abandoned {ex['no result']}; curtailed or anomalous innings {audit['anomalous_matches_dropped']['total_unique']}"],
    ["<b>Analysed</b>", f"<b>{audit['matches_kept']}</b>", "425 T20I + 353 IPL matches; 2022: 169, 2023: 128, 2024: 169, 2025: 179, 2026: 133"],
    ["Deliveries", f"{audit['balls_kept']:,}", "Innings 1 and 2 only; 816 batters and 589 bowlers appear"]], [4.2, 1.8, 11.4]), Spacer(1, 6)]

# ================================================================== 5. PRESSURE INDEX
st.append(H1("5. Defining and detecting pressure"))
st.append(P("The Pressure Index combines three ideas: how <i>close</i> the contest is, how much each ball can still <i>swing</i> it (leverage), and how <i>stressed</i> each side is relative to where it should be. All three are computed from a win-probability engine that we build and validate ourselves."))
st.append(H2("5.1 Expected runs-to-go and win probability"))
st += EQ("μ(b, w) = expected runs still to be scored with b balls left and w wickets lost      σ(b, w) = residual standard deviation")
st.append(P(f"μ is a monotone gradient-boosted model (non-decreasing in balls left, non-increasing in wickets lost) trained on innings-1 runs-to-go with competition-type and season terms; σ comes from cross-fitted squared residuals. <b>Chase:</b> with R runs required, the chasing side's win probability is p = Φ((μ − R + 0.5)/σ), recalibrated by a logistic (Platt) layer fitted on chase deliveries (intercept {dg['platt_chase'][0]:.2f}, slope {dg['platt_chase'][1]:.2f}); terminal states are exact; revised targets are excluded. <b>Innings 1:</b> p = Φ((score + μ − par)/√(σ² + {dg['sd_first_innings_total']:.1f}²)), where par is the mean first-innings total of the competition-season (T20I 168-181, IPL 171-195 across 2022-26) and {dg['sd_first_innings_total']:.1f} is the spread of totals; Platt-calibrated (intercept {dg['platt_inn1'][0]:.2f}, slope {dg['platt_inn1'][1]:.2f})."))
st.append(P("<b>Validation (five-fold, grouped by match).</b>"))
vc, vi = dg["wp_cv_chase"], dg["wp_cv_inn1"]
st += [TBL([["Model", "Deliveries", "AUC", "Brier", "Brier: constant", "Brier: RRR-only", "Log-loss"],
            ["Chase win probability", f"{vc['n_rows']:,}", f"{vc['auc_cal']:.3f}", f"{vc['brier_cal']:.3f}", f"{vc['brier_const']:.3f}", f"{vc['brier_rrr_only']:.3f}", f"{vc['logloss_cal']:.3f} (RRR-only {vc['logloss_rrr_only']:.3f})"],
            ["Innings-1 win probability", f"{vi['n_rows']:,}", f"{vi['auc_cal']:.3f}", f"{vi['brier_cal']:.3f}", f"{vi['brier_const']:.3f}", "n/a", f"{vi['logloss_cal']:.3f} (constant {vi['logloss_const']:.3f})"]], [4.0, 1.9, 1.2, 1.2, 2.2, 2.2, 4.7]), Spacer(1, 3),
       P("The chase model clearly beats a required-rate rule because it uses wickets in hand and balls left jointly. The innings-1 model is, unsurprisingly, only moderately informative (AUC 0.70): first-innings win probability barely moves ball to ball, which is why innings 1 needs the explicit stress terms in 5.3.", "body")]
st.append(FIG("fig/f4_calibration.png", 12.5, "Figure 1. Out-of-fold calibration of the two win-probability models (12 equal-count bins)."))
st.append(H2("5.2 Chases (innings 2)"))
st += EQ("LI = E|ΔWP| over next-ball outcomes {0,1,2,3,4,6,W} ÷ mean(E|ΔWP|)      L = 1 − exp(−LI/2)")
st += EQ("D<sub>bat</sub> = (1 − p)·min(1, 4p)      D<sub>bowl</sub> = p·min(1, 4(1 − p))      PI<sub>raw</sub> = 0.6·L + 0.4·D")
st.append(P("<b>Leverage L</b> is the expected absolute swing in win probability from the next delivery, using the empirical outcome distribution by innings, over and wickets-lost group. It combines required rate, balls left and wickets in one number with no hand-set weights, and it keeps its absolute scale (a rank transform would flatten the difference between a tight over 19 and a quiet over 4). <b>Stress D</b> is 'behind but still alive' for the batter (high when the chase is failing but not hopeless; it collapses to zero when the match is effectively over) and the mirror image for the bowler (the batting side is ahead and has not yet won)."))
st.append(H2("5.3 First innings (setting a target)"))
st += EQ("PI<sub>raw</sub> = 0.40·φ(over) + 0.35·D + 0.25·C")
st += BL([f"<b>φ(over): end-of-innings intensity.</b> Runs per over by over, min-max scaled to [0,1] (data-driven: 0 in over 1, about 0.5 in overs 3-6, a drop to 0.18 in over 7, then a steady climb to 0.75 in over 19 and 1.0 in over 20).",
          "<b>D: below-par stress.</b> The same 'behind but alive' term, using the innings-1 win probability (projected score against competition-season par).",
          "<b>C: collapse index.</b> The empirical CDF of wickets lost at this over of the innings (a value near 1 means an unusually early collapse, e.g. three down in the powerplay). For bowlers we use 1 − C: the batting side has wickets in hand."])
st.append(H2("5.4 Stakes, mapping and thresholds"))
st += BL(["<b>Stakes.</b> PI<sub>raw</sub> is multiplied by (1 + 0.2·s) with s = 0 for league and bilateral matches, 0.5 for Super 4 and 1 for Qualifier, Eliminator, Semi-final and Final. Knockout matches are too few to estimate this weight from the data, so it is stated as a prior and tested at 0, 0.1 and 0.4 (Section 9).",
          "<b>Percentile mapping within innings.</b> PI is the percentile rank of PI<sub>raw</sub> within each innings, so each innings type contributes a quarter of its balls to the high-pressure set (innings 1 and 2 are not on a common raw scale). A pooled mapping is a sensitivity variant.",
          "<b>Classification.</b> <b>High pressure</b> = PI ≥ 0.75 (44,636 batter-balls and 46,538 bowler deliveries); <b>normal</b> = PI ≤ 0.50; the middle band is excluded from the differential. 0.70 and 0.80 cut-offs are sensitivity variants."])
st.append(FIG("fig/f1_pi_shape.png", 16, "Figure 2. (a) Share of balls flagged high-pressure by over. (b) In chase overs 16-20, PI is high only while the match is live; decided games (win probability near 0 or 100%) are not flagged."))
st.append(FIG("fig/f2_chase_anatomy.png", 14.5, "Figure 3. Worked example: win probability (top) and PI (bottom) through a close league-stage chase. The bowling side's PI rises as the chasing side moves ahead; the batting side's PI stays high while it is behind but alive."))
hp_inn = Pb.innings.value_counts(normalize=True); hp_ph = Pb.phase.value_counts(normalize=True); np_ph = Nb.phase.value_counts(normalize=True)
st.append(H2("5.5 Evidence that PI separates pressure from ordinary play"))
st.append(P(f"The high-pressure set is {100*hp_inn[1]:.0f}% innings 1 and {100*hp_inn[2]:.0f}% innings 2, spread over {100*hp_ph['Powerplay']:.0f}% powerplay, {100*hp_ph['Middle']:.0f}% middle overs and {100*hp_ph['Death']:.0f}% death overs (the normal set is {100*np_ph['Powerplay']:.0f}% / {100*np_ph['Middle']:.0f}% / {100*np_ph['Death']:.0f}%), and {100*(Pb.ctype=='T20I').mean():.0f}% T20I and {100*(Pb.ctype=='IPL').mean():.0f}% IPL, closely mirroring the mix of the data. Knockout and Super 4 balls make up about 7% of it. Raw league-wide outcomes differ in the expected direction: batters score {lg['sr_p']:.0f} runs per 100 balls under pressure versus {lg['sr_n']:.0f} in normal play, are dismissed {lg['out_p']:.1f}% versus {lg['out_n']:.1f}% of balls, and face {lg['dot_p']:.0f}% versus {lg['dot_n']:.0f}% dot balls; bowlers concede {lg['ec_p']:.1f} versus {lg['ec_n']:.1f} runs per over (these raw gaps include the death-heavy composition of the pressure set, which is exactly why ratings use phase-adjusted residuals; the average situation-adjusted pressure effect is +{perm['bat']['mean_delta']:.1f} runs/100 balls for batters and {perm['bowl']['mean_delta']:.1f} for bowlers)."))
st.append(FIG("fig/f3_dose_response.png", 16.2, "Figure 4. League-wide batter outcomes by PI decile (raw, unadjusted). Scoring and boundaries dip and dot balls rise through the middle-to-upper deciles; in the top decile batters attack again (about 8.7 runs per over) at the cost of a much higher dismissal rate (8.0% against about 5% elsewhere)."))


# ================================================================== 6. PERFORMANCE
st.append(H1("6. Measuring performance against baselines"))
st.append(H2("6.1 Situation-expected baseline"))
st.append(P("For every ball faced (wides excluded) and every ball bowled, we fit gradient-boosted context models for expected bat runs (bowler: runs conceded, wides and no-balls included; byes and leg-byes not charged), P(dot), P(boundary) and P(dismissal) using only over, wickets lost, innings, competition type, season and free-hit status. This gives each ball an expected outcome for an average player, so a death specialist is compared with the average death ball, not with the powerplay."))
st.append(H2("6.2 Opposition and competition adjustment"))
st.append(P(f"For each outcome, residuals are decomposed with a ridge (BLUP) model into own-player, opposing-player and opposing-team effects; the penalty is set from the data's own between-player variance (e.g. prior sample size ≈ 560 balls for batter run residuals). The opposition part is subtracted from each ball, so a batter facing elite bowlers, or a bowler facing weaker T20I sides, is credited accordingly. Competition and season enter the baseline (first-innings par rises from 168-181 in T20Is to 171-195 in the IPL, including a 2023 impact-player-era step), so IPL and T20I performances are measured against their own scoring environments."))
st.append(H2("6.3 One currency: runs above expectation, wickets priced"))
st += EQ("V<sub>bat</sub> = (r − r̂ − opp<sub>r</sub>) − λ<sub>w</sub>·(out − ôut − opp<sub>out</sub>)       V<sub>bowl</sub> = −(runs conceded − ĉ − opp) + λ<sub>w</sub>·(wkt − ŵkt − opp)")
st.append(P("λ<sub>w</sub> = μ(b−1, w) − μ(b−1, w+1) is the run cost of losing a wicket in that state (larger early in an innings). This answers the SR-versus-average trade-off with a principled exchange rate instead of an arbitrary weight: a batter who scores quickly but gets out too often is charged for it, in runs. Dismissals credit the striker (run-outs of the striker included, retire-outs and non-striker run-outs excluded); bowler wickets are caught, bowled, LBW, stumped and hit-wicket only."))
st.append(H2("6.4 The six rating components and how they are balanced"))
ew = lambda k: f"{100*eff['bat'][k]:.0f}% / {100*eff['bowl'][k]:.0f}%"
st += [TBL([["Component (higher = better)", "Batting version", "Bowling version", "Nominal weight", "Effective share (bat / bowl)"],
            ["R: scoring / run prevention", "Runs above expectation per 100 balls", "Runs saved per 100 deliveries", "25%", ew("R")],
            ["Wsave: wicket value", "Dismissal avoidance, in runs", "Wicket-taking, in runs", "15%", ew("Wsave")],
            ["Δ: clutch differential", "Pressure value minus own normal-situation value", "Same", "20%", ew("Delta")],
            ["DOT: dot-ball control", "Fewer dot balls than expected (pp)", "More dot balls than expected (pp)", "15%", ew("DOT")],
            ["BD: boundaries", "More boundaries than expected (pp)", "Fewer boundaries conceded (pp)", "10%", ew("BD")],
            ["CONS: consistency", "Share of pressure innings with positive net value", "Share of pressure spells with positive net value", "15%", ew("CONS")]],
           [3.9, 4.4, 4.0, 1.9, 3.2]), Spacer(1, 3),
       P("Nominal weights express intent (value 40%, clutch gap 20%, control 25%, consistency 15%). <b>Effective share</b> is each component's actual share of the variance of the final composite after shrinkage. Where the data cannot separate a component from noise (wicket-taking and clutch gap for bowlers, where the estimated true between-player spread is zero), shrinkage drives its contribution to roughly nothing instead of letting noise decide the ranking. Strike rate and average are not scored separately because the run-and-wicket currency already prices both; dot % and boundary % are retained as secondary control measures with smaller weights.", "cap")]


# ================================================================== 7. FAIRNESS
st.append(H1("7. Fair comparisons: sample size, shrinkage, context"))
st.append(H2("7.1 Eligibility"))
st.append(P(f"A player is rated only if he has at least <b>100 balls</b> (batters) or <b>100 deliveries</b> (bowlers) in high-pressure situations, in at least <b>15 matches</b>, plus at least 100 balls in normal situations (needed for the differential). That leaves {len(fb)} of 816 batters and {len(fw)} of 589 bowlers. The gate is deliberately a floor, since shrinkage does the main work; ranks are re-computed at 80/12 and 150/20 (Section 9). A 20-ball, SR-250 cameo is never rated at all, and even at the gate it would be heavily shrunk (below)."))
st.append(H2("7.2 Empirical-Bayes shrinkage"))
st += EQ("θ̂<sub>i</sub> = μ<sub>0</sub> + B<sub>i</sub>(y<sub>i</sub> − μ<sub>0</sub>),   B<sub>i</sub> = τ² / (τ² + se<sub>i</sub>²),   τ² from DerSimonian-Laird,  se<sub>i</sub> clustered by match")
st.append(P(f"Each component is shrunk separately (the noisy wicket term must not drag down the reliable run signal). Standard errors are cluster-robust by match because balls within an innings are correlated. Under the batter scoring component, τ ≈ {tauR:.1f} runs per 100 balls, so a batter's own data receives weight B = {Bn[20]:.2f} at 20 pressure balls, {Bn[100]:.2f} at 100, {Bn[300]:.2f} at 300 and {Bn[600]:.2f} at 600. The 20-ball SR-250 outlier is therefore pulled almost entirely to the mean, while a proven performer's estimate barely moves. Composite scores are expressed in talent-SD units, (θ̂ − μ<sub>0</sub>)/τ, which is stable even as τ approaches zero because B shrinks with it."))
st.append(H2("7.3 Two estimators, chosen by predictive accuracy"))
st += BL(["<b>Estimator A (pressure-only):</b> each pressure estimate is shrunk to the population mean of pressure performance.",
          "<b>Estimator B (hierarchical):</b> true pressure ability θ<sub>P</sub> = θ<sub>N</sub> + δ, where θ<sub>N</sub> is normal-situation ability and δ is the clutch deviation. Posterior means use both samples, borrowing strength from the much larger normal sample; δ's variance is estimated, not assumed.",
          "<b>Final rating = 50/50 blend</b> of the two composites (z-scored)."])
st.append(P(f"We compared estimators out of sample: in 100 random half-splits of matches we estimated players on one half and correlated the estimate with their raw pressure value in the other half. Estimator B is best (batters {pv['bat'].val_B_hierarchical.mean():.3f}, bowlers {pv['bowl'].val_B_hierarchical.mean():.3f}) and beats A in 69-72% of splits; raw statistics are far worse ({pv['bat'].val_raw_pressure.mean():.3f}, {pv['bowl'].val_raw_pressure.mean():.3f}). The blend is statistically indistinguishable from B ({bv['bat']['val_0.5'].mean():.3f}, {bv['bowl']['val_0.5'].mean():.3f}) across a flat blend-weight plateau (0.5 to 1.0), but keeps more weight on direct pressure evidence. For example, one star batter rated 18th has a raw pressure value of −51 runs/100 balls on just 115 balls; the shrunk estimate treats this as the noise it most likely is. The low absolute correlations are themselves a finding: pressure performance is only weakly persistent from match to match."))
st.append(FIG("fig/f5_predictive.png", 14, "Figure 5. Out-of-sample validation (100 random match splits, players with ≥40 pressure balls in each half). Whiskers: 10th-90th percentile across splits."))
st.append(FIG("fig/f6_shrinkage.png", 14, "Figure 6. Raw versus shrunk pressure value. Small samples (light points) are pulled hardest towards the mean; large-sample performers stay near the diagonal."))
st.append(H2("7.4 Volume, IPL versus T20I, opposition"))
st += BL([f"<b>Volume bias.</b> Ratings are per-ball residuals, so volume earns nothing directly. The rank correlation between the composite and pressure balls faced is {vol_b:.2f} for batters and {vol_w:.2f} for bowlers; the top 10 are not simply the highest-volume players (pressure balls range from 157 to 576 among the top-10 batters and from 200 to 571 among bowlers).",
          "<b>IPL versus T20I.</b> One pooled model with competition-by-season baselines; PI is defined identically in both (percentile within innings), so each competition contributes in proportion to its own pressure balls. Ratings restricted to IPL rows or to T20I rows correlate with the pooled ranking at Kendall τ = 0.75 / 0.70 (batters) and 0.72 / 0.69 (bowlers), so competition context matters and we report this openly (Section 9).",
          "<b>Opposition strength.</b> Opposing-player and opposing-team effects are removed from every ball (Section 6.2). The stakes multiplier covers knockout context; bilateral dead rubbers are not separately identifiable from the supplied files."])


# ================================================================== 8. RESULTS / EVIDENCE
st.append(H1("8. Evidence for the top 10 and how the order was set"))
st.append(P("The base model is the 50/50 blend of A and B with the specification above. To remove dependence on any single design choice, we re-ran the entire rating under 18 specifications (base; high-pressure cut-off 0.70 and 0.80; stakes multiplier 0, 0.1, 0.4; leverage weight 0.4 and 0.8; pooled PI mapping; phase-neutral PI; equal component weights; value-only; no clutch differential; differential-heavy; gates 80/12 and 150/20; estimator A only; estimator B only). The <b>final rank is the mean rank across these 18 specifications</b>, with ties broken by the base composite. IPL-only and T20I-only runs are reported separately because they change the player pool."))
st.append(FIG("fig/f7_rank_stability.png", 15.6, "Figure 7. Each dot is a player's rank under one of the 18 specifications; the orange tick is the median. Shaded area: top 10."))
st.append(FIG("fig/f8_components.png", 15.6, "Figure 8. What drives each top-10 rating: weighted component contributions relative to the average eligible player."))

def evidence(side):
    r = (fb if side == "bat" else fw).head(15)
    if side == "bat":
        d = [["#", "Player", "ID", "Press. balls", "Raw value", "Final value", "Δ gap", "Base rank", "Mean rank (18)", "Best-worst", "Boot. P(top-10)", "Boot. rank 5-95%"]]
        for pid, x in r.iterrows(): d.append([int(x.final_rank), x["name"], pid, int(x.nP), f"{x.Value_raw:+.0f}", f"{x.Value_shr:+.0f}", f"{x.Delta_shr:+.1f}", int(x.base_rank), f"{x.consensus_mean_rank:.1f}", f"{int(x.best_rank)}-{int(x.worst_rank)}", f"{x.p_top10:.2f}", f"{int(x.boot_rank_lo)}-{int(x.boot_rank_hi)}"])
    else:
        d = [["#", "Player", "ID", "Press. balls", "Raw value/over", "Final value/over", "Base rank", "Mean rank (18)", "Best-worst", "Boot. P(top-10)", "Boot. rank 5-95%"]]
        for pid, x in r.iterrows(): d.append([int(x.final_rank), x["name"], pid, int(x.P_balls), f"{x.Value_raw*0.06:+.2f}", f"{x.Value_shr*0.06:+.2f}", int(x.base_rank), f"{x.consensus_mean_rank:.1f}", f"{int(x.best_rank)}-{int(x.worst_rank)}", f"{x.p_top10:.2f}", f"{int(x.boot_rank_lo)}-{int(x.boot_rank_hi)}"])
    return d
bd = evidence("bat"); bwd = evidence("bowl")
st += [KeepTogether([H2("Batters: ranks 1-15"), TBL(bd, [0.6, 2.7, 1.0, 1.1, 1.2, 1.3, 1.0, 1.2, 1.5, 1.4, 1.6, 1.8])]), Spacer(1, 3),
       P("Value is runs above expectation per 100 balls (wicket-adjusted). 'Raw value' is before opposition-aware shrinkage; Δ gap is the shrunk pressure-minus-normal differential. Bootstrap: 300 resamples of matches (base specification).", "cap"),
       KeepTogether([H2("Bowlers: ranks 1-15"), TBL(bwd, [0.6, 3.0, 1.0, 1.2, 1.5, 1.6, 1.1, 1.4, 1.5, 1.5, 1.8])]), Spacer(1, 3),
       P("Value is runs saved per over including the run value of wickets, against expectation in the same situation and opposition. Pressure balls are legal deliveries.", "cap")]
st.append(H2("Reading the results"))
nm = lambda r: fb[fb.final_rank == r].iloc[0]; nw = lambda r: fw[fw.final_rank == r].iloc[0]
st += BL([
    f"<b>Batters.</b> Abhishek Sharma leads on {int(nm(1).nP)} pressure balls across {int(nm(1).P_matches)} matches (raw SR {nm(1).P_SR:.0f}) and is first or third in every specification. Suryakumar Yadav ({int(nm(2).nP)} balls, the largest pressure sample in the group) and Klaasen are the most reliable on volume-supported evidence. Salt, Naman Dhir, Neesham and Patidar have the smallest pressure samples in the top 10 (157-192 balls); their ratings are shrunk accordingly, and Patidar and Dhir remain high regardless. Russell's rating combines elite boundary hitting with a high dot-ball share (boom-and-bust), so his place is more specification-dependent.",
    f"<b>Places 7-10 are close calls.</b> Salt, Russell, Iyer, Neesham, Fakhar Zaman and Kohli are separated by under two mean ranks (10.4 to 12.2), and no one of them is in the top 10 in more than 10 of 18 specifications; the batter bootstrap also assigns Fakhar Zaman the highest chance of a top-10 finish (0.76) on the base specification. We resolved this by the pre-declared consensus rule rather than by judgement.",
    f"<b>Bowlers.</b> Narine and Bumrah hold ranks 1-2 in every specification, and Pathirana is third in most (never worse than 6th). Chakravarthy, Theekshana, Nawaz, Ellis, Rashid Khan and Axar Patel complete a robust top 9 (each in the top 10 in 13-18 of 18 specifications); Hazlewood narrowly takes the 10th place ahead of Hasaranga (mean ranks 11.7 vs 12.5).",
    "<b>Where familiar names landed.</b> Several players with strong reputations fall outside the top 10 on the data. Kohli is 12th (pressure SR 157, average 48, but a smaller shrunk pressure edge than the top group); Jaiswal 16th; Gill 22nd; Pooran 28th; Buttler 26th (raw pressure value +2, shrunk +22 because of his strong normal-situation record); Rohit Sharma 95th and Pant 107th. Among bowlers, Rabada is 36th, Bhuvneshwar 31st, Arshdeep 27th (largest pressure workload in the data, 947 deliveries, but only slightly better than expectation), Siraj 78th and Jadeja 23rd. These are statements about measured, situation-adjusted pressure performance in this dataset, not about overall quality."])


# ================================================================== 9. RELIABILITY
st.append(H1("9. Reliability, robustness and validation"))
sb, sw = sens["bat"]["summary"].set_index("variant"), sens["bowl"]["summary"].set_index("variant")
rows = [["Specification changed", "Batters: eligible", "Kendall τ", "Top-10 overlap", "Bowlers: eligible", "Kendall τ", "Top-10 overlap"]]
for v in sb.index[1:]:
    rows.append([v, int(sb.loc[v, "n_elig"]), f"{sb.loc[v, 'kendall_tau_vs_base']:.2f}", f"{int(sb.loc[v, 'top10_overlap'])}/10", int(sw.loc[v, "n_elig"]), f"{sw.loc[v, 'kendall_tau_vs_base']:.2f}", f"{int(sw.loc[v, 'top10_overlap'])}/10"])
st += [TBL(rows, [5.4, 1.9, 1.5, 1.9, 1.9, 1.5, 1.9], zebra=True), Spacer(1, 3),
       P("Kendall τ and top-10 overlap are measured against the base specification (before consensus). IPL-only and T20I-only runs use relaxed gates (60 balls, 8 matches) and different player pools.", "cap")]
ts_b, ts_w = sens["bat"]["time_split"], sens["bowl"]["time_split"]
st += BL([
    f"<b>Bootstrap by match (300 resamples).</b> Rank intervals are wide: even the first-ranked batter's 5-95% rank interval runs to 62, and the highest P(top-10) is {final['bat']['res'].p_top10.max():.2f} (batters) and {final['bowl']['res'].p_top10.max():.2f} (bowlers). With 100-600 pressure balls per player this is the honest limit of what ball-by-ball data can resolve, and it is why we rank by a robust consensus rather than a single fitted score.",
    f"<b>Out-of-sample prediction.</b> Shrunk ratings predict held-out pressure performance 1.6-2.0x better than raw statistics (Figure 5).",
    f"<b>Time stability.</b> Ratings from 2022-23 and 2024-26 matches (each half has small samples) correlate at Spearman {ts_b['spearman']:.2f} (n = {ts_b['n']} batters) and {ts_w['spearman']:.2f} (n = {ts_w['n']} bowlers). Bowlers are more persistent; batters less so.",
    f"<b>Permutation test for clutch.</b> Shuffling pressure labels within each match-innings-phase cell (300 permutations) gives a null variance of standardised pressure-minus-normal gaps of {perm['bat']['null_mean']:.2f} (batters) and {perm['bowl']['null_mean']:.2f} (bowlers) versus observed {perm['bat']['obs_var_z']:.2f} and {perm['bowl']['obs_var_z']:.2f} (p = {perm['bat']['p_value']:.2f} and {perm['bowl']['p_value']:.3f}). Individual clutch gaps are indistinguishable from chance at this sample size, which is why they carry little effective weight.",
    "<b>Sanity checks.</b> No innings exceeds 120 legal balls or 10 wickets after cleaning; 99% of losing chases finish below their reconstructed target; mean value across all balls is near zero by construction; the opposition-effect SDs are small and plausible (about 0.08 runs per ball for batter-facing bowlers' run effects)."])
st.append(H1("10. Limitations"))
st += BL([
    "<b>Sample size is the binding constraint.</b> Pressure performance is weakly persistent (out-of-sample correlations 0.12-0.20), rank intervals are wide, and places 7-10 among batters are close calls. We say so and quantify it.",
    "<b>Pressure is a model, not an observation.</b> The first-innings win-probability model is only moderately informative (AUC 0.70); first-innings PI therefore rests on explicit stress terms whose weights (0.40/0.35/0.25) are judgemental and tested by sensitivity, and its end-of-innings term makes first-innings death overs high-pressure by construction (partly offset because the baseline conditions on the over, and by the phase-neutral variant).",
    "<b>Extras runs are reconstructed</b> (error about 1.5 runs in match margins), which adds small noise to required rates. No pitch, weather, dew, fielding, bowling-plan or ball-tracking data are available; role and batting position are not modelled.",
    "<b>Selection effects.</b> Captains choose who bats and bowls at key moments, so pressure exposure is not random. Situation-matched baselines reduce but do not remove this (e.g. specialist death bowlers).",
    "<b>Stakes weighting is a prior</b>, not an estimate; series-decider and qualification-stake context are not identifiable from the files.",
    "<b>Clutch skill is not identified.</b> We rate pressure-state performance, not an innate trait; the pressure-minus-normal differential is statistically weak here.",
    "<b>Pool differences.</b> T20I and IPL player pools overlap only partly; IPL-only and T20I-only ratings agree moderately (τ ≈ 0.7), so cross-competition comparisons carry extra uncertainty. Emerging Teams events, rain-affected matches and super overs are excluded."])

# ================================================================== APPENDICES


# ================================================================== 11. RAJASTHAN ROYALS
rrt = pd.read_pickle("rr_table.pkl"); gain = pd.read_pickle("rr_gain.pkl").set_index("name"); prof = pd.read_pickle("rr_profiles.pkl").set_index("name")
st.append(H1("11. Rajasthan Royals: how the 20 players could strengthen the squad for the IPL 2027 auction"))
st += CALL("<b>Status of the facts (17 November 2026).</b> The IPL 2027 retention and release deadline is today; the sources available when this report was written had not yet published any franchise's official list, so RR's retained and released players are <b>unconfirmed</b>. We therefore use RR's 2026 squad (27 players, nine overseas) as the baseline and treat every player not already at RR as available <i>only if his franchise releases or trades him</i>. Purse, auction date and prices are not modelled.", ORANGE)
st.append(P("<b>Context.</b> RR finished fourth in the 2026 league stage (Kumar Sangakkara coach, Riyan Parag captain), beat Sunrisers Hyderabad by 47 runs in the Eliminator and then lost to Gujarat Titans in the next qualifier (from the supplied match data). Their 2026 squad was reshaped by the trade that sent Sanju Samson to Chennai in exchange for Ravindra Jadeja and Sam Curran, and by a Ravi Bishnoi purchase at the auction. Reports also link RR with a multi-player trade with Lucknow involving Curran and Mohammed Shami."))
st.append(H2("11.1 Method: the same lens, applied to RR"))
st += BL(["<b>Where is RR strong or weak under pressure?</b> We apply the identical Pressure Index and per-ball value (Sections 5-6) to every IPL team's high-pressure balls in 2025-26 and in 2026 alone, by phase, batting slot and bowling type, and rank RR among the 10 teams.",
          "<b>What does each of the 20 players offer?</b> Role and phase profile from his own pressure balls, IPL workload per season, 2026 franchise and eligibility.",
          "<b>How much could he add?</b> Indicative gain = (player's final value − the average final value of RR's rated players in that role) × his typical IPL pressure workload per 14-match season. It is a central estimate with wide uncertainty, not a forecast: it ignores price, overseas-slot limits, injuries and batting-order context."])
st.append(H2("11.2 RR's pressure profile against the league (IPL, high-pressure balls)"))
def rr_rows():
    d = [["Segment", "RR value 2025-26 (± 1 SE)", "League average", "RR rank of 10: 2025-26", "RR rank of 10: 2026", "RR pressure balls (2025-26)"]]
    for _, r in rrt.iterrows():
        unit = "runs/100 balls" if r.disc == "Batting" else "runs/over"
        fmt = (lambda v: f"{v:+.0f}") if r.disc == "Batting" else (lambda v: f"{v:+.2f}")
        bold = lambda k: f"<b>{int(k)}</b>" if k >= 7 else f"{int(k)}"
        d.append([f"{r.disc}: {r.label}", f"{fmt(r.rr)} ± {abs(r.se):.{0 if r.disc=='Batting' else 2}f} {unit}", fmt(r.league), bold(r['rank']), bold(r.rank26), int(r.n)])
    return d
st += [TBL(rr_rows(), [4.4, 4.2, 2.1, 2.3, 2.2, 2.2]), Spacer(1, 3),
       P("Value is the same situation- and opposition-adjusted currency as the player ratings (batting: runs above expectation per 100 balls; bowling: runs saved per over, wickets priced). Rank 1 is best; ranks of 7 or worse are in bold. <b>Read these as directional, not conclusive:</b> a season of one team's pressure balls is small, and no single gap is more than about one standard error. We trust the pattern only because it repeats across the 2025-26 and 2026-only windows.", "cap")]
st += BL(["<b>Strengths.</b> The top order (slots 1-2) is first or second of ten, bowling in the middle overs is second, and spin is third or second. Jaiswal (rated 16th of 111 batters) and the very promising Vaibhav Suryavanshi (116 pressure balls, too few to rate, but +66 runs per 100 balls raw) anchor this.",
          "<b>Weaknesses.</b> Death-overs batting (8th, then 6th), death-overs bowling (7th in both windows), pace bowling under pressure (5th, then 7th) and batters in slots 3-5 (7th, then 6th). RR bowled 433 of its 885 pressure pace deliveries in 2025-26 at the death, so the death-bowling weakness matters most."])
st.append(H2("11.3 Where RR's current players stand in our ratings"))
st.append(P("Our model rates (≥100 pressure balls, ≥15 matches) batters Jaiswal 16th, Shanaka 47th, Parag 61st, Jadeja 71st, Jurel 78th, Hetmyer 97th and Curran 98th of 111; and bowlers Jadeja 23rd, Bishnoi 35th, Archer 38th, Deshpande 43rd, Sandeep Sharma 70th and Curran 89th of 112. Most other RR players have too few pressure balls to rate. No RR player is in our top 10 in either list. The averages used as the benchmark for the gains below are: batting slots 1-2 +25, slots 3-5 +12.5, slots 6+ +10 runs per 100 balls; pace bowling −0.04 and spin +0.29 runs per over."))
st.append(H2("11.4 The 20 players: availability, fit and indicative gain"))
def gcell(n):
    if n not in gain.index or n in ("Andre Russell", "Mohammad Nawaz", "Jimmy Neesham"): return "n/a"
    return f"{gain.loc[n,'gain']:+.0f}"
bf = lambda n, rk: f"<b>{n}</b> (#{rk})"
FIT = [
 # name, rank, side, availability, profile, need, verdict
 ("Abhishek Sharma", 1, "SRH. Contracted; needs release or trade.", "Left-hand opener; 78% of his pressure balls in the powerplay.", "Low (slots 1-2 already 1st)", "Best batter in the data but the least needed: a luxury, or insurance if Jaiswal or Suryavanshi are unavailable."),
 ("Suryakumar Yadav", 2, "MI. Contracted; trade is the realistic route.", "No. 3-4; balanced across powerplay (41%), middle (42%) and death (16%).", "Medium-high (slots 3-5: 7th/6th)", "Largest batting gain (+18). Indian player, so no overseas slot; fits the middle-order gap."),
 ("Heinrich Klaasen", 3, "SRH. Contracted. Overseas.", "No. 5; 52% middle overs, 28% death; wicketkeeper.", "Medium-high", "Strong middle-overs power and a keeper option beside Jurel; competes for an overseas place with Hetmyer and Ferreira."),
 ("Rajat Patidar", 4, "RCB. Contracted (RCB captain in 2025).", "No. 3-4; 46% powerplay, 44% middle.", "Medium", "Efficient Indian middle-order option; unlikely to be released."),
 ("Naman Dhir", 5, "MI. Contracted.", "No. 5; 45% of pressure balls at the death; 142 IPL pressure balls, a smaller sample.", "High (death batting 8th/6th)", "Indian finisher profile that matches RR's death hole; cheaper bet, but the rating carries the most sampling risk."),
 ("Tim David", 6, "RCB. Contracted. Overseas.", "No. 6; 56% at the death; 270 IPL pressure balls.", "High", "Best pure finishing fit (+16); competes for an overseas slot with Hetmyer (rated 97th)."),
 ("Phil Salt", 7, "RCB. Contracted. Overseas.", "Opener; 61% powerplay.", "Low", "Not needed with Jaiswal and Suryavanshi: skip."),
 ("Andre Russell", 8, "Retired from the IPL in Nov 2025 (now a KKR power coach).", "No. 7; 64% at the death.", "n/a", "<b>Cannot be signed.</b> Kept in the data-driven ranking, but his role is better filled by David or Dhir."),
 ("Shreyas Iyer", 9, "PBKS. Contracted.", "No. 4; 36% powerplay, 44% middle.", "Medium", "Reliable Indian anchor (+9) but overlaps with Parag in the same slot."),
 ("Jimmy Neesham", 10, "In no IPL 2026 squad in the data.", "No. 6; 60% middle overs; only 2 IPL matches, so the rating rests almost entirely on non-IPL T20 data.", "n/a", "Low-cost depth idea but the evidence does not transfer to IPL conditions: treat with caution."),
 ("Sunil Narine", 1, "KKR. Contracted. Overseas.", "Spin; 24% powerplay, 56% middle.", "Low-medium (spin 3rd/2nd)", "Elite (+1.3 runs/over) but spin is RR's strength, so the marginal gain (+15) is smaller than for pace."),
 ("Jasprit Bumrah", 2, "MI. Contracted.", "Pace; 47% powerplay, 37% death; ~23 pressure overs per season.", "High (pace 5th/7th; death 7th)", "<b>Largest gain of all 20 (+24)</b>; Indian, so no overseas slot. Realistic only through a trade; likely the costliest."),
 ("Matheesha Pathirana", 3, "KKR; reported ₹18 crore buy at the 2026 auction. Overseas.", "Pace; 66% of pressure deliveries at the death.", "High", "Death specialist (+14); RR has used this type before. Price is the question."),
 ("Varun Chakravarthy", 4, "KKR. Contracted.", "Leg-spin; 55% middle overs.", "Low", "RR is already strong in spin (Jadeja, Bishnoi); low priority (+4)."),
 ("Maheesh Theekshana", 5, "Released by RR before the 2026 auction and unsold there; likely back in the pool. Overseas.", "Off-spin; 41% powerplay, 32% middle.", "Low-medium", "Cheapest realistic route, but spin is not RR's need (+7); powerplay spin cover at best."),
 ("Mohammad Nawaz", 6, "Pakistan player: Pakistani players have not featured in the IPL since the first season.", "Left-arm spin; 70% middle overs; no IPL data.", "n/a", "<b>Cannot be signed</b> in practice."),
 ("Nathan Ellis", 7, "CSK. Contracted. Overseas.", "Pace; 51% death; small IPL sample (12 matches).", "High", "Strong death profile (+17); competes with Archer, Burger, Milne and Maphaka for an overseas place."),
 ("Rashid Khan", 8, "GT. Contracted. Overseas.", "Leg-spin; 51% middle overs.", "Low", "Elite, but RR has Bishnoi and Jadeja (+4); low priority."),
 ("Axar Patel", 9, "DC. Contracted.", "Left-arm spin; 53% middle overs.", "Low", "Duplicates Jadeja's role (+3); low priority."),
 ("Josh Hazlewood", 10, "RCB. Contracted. Overseas.", "Pace; 50% powerplay, 38% death.", "Medium-high", "Powerplay plus death control (+10); another overseas pace body for RR's crowded group."),
]
rowsf = [["Player (rank)", "2026 franchise / availability", "Profile from his pressure balls", "RR need", "Gain (runs per season)", "Verdict for RR"]]
for n, rk, av, pf, need, vd in FIT:
    rowsf.append([bf(n, rk), av, pf, need, gcell(n), vd])
st += [P("Batters first, then bowlers. 'Gain' is the indicative runs per 14-match season over RR's average rated player in the same role (about 10 runs is roughly one win in T20, so +10 to +25 is one to two and a half wins)."), TBL(rowsf[:11], [2.5, 3.0, 3.4, 2.0, 1.5, 5.0]), Spacer(1, 6), TBL(rowsf[:1] + rowsf[11:], [2.5, 3.0, 3.4, 2.0, 1.5, 5.0]), Spacer(1, 4)]
st.append(H2("11.5 Recommended shortlist for RR (if released or tradable)"))
st += BL(["<b>Death-overs pace (priority 1):</b> Jasprit Bumrah (+24) > Matheesha Pathirana (+14) and Nathan Ellis (+17) > Josh Hazlewood (+10). This addresses RR's weakest and highest-volume pressure area.",
          "<b>Finishing and slots 3-5 (priority 2):</b> Tim David (+16, overseas) and Naman Dhir (+10, Indian, riskier sample) for the death; Suryakumar Yadav (+18) and Klaasen (+11) for the middle order.",
          "<b>Top order and spin (no priority):</b> Abhishek Sharma, Phil Salt, Narine, Chakravarthy, Rashid Khan, Axar Patel and Theekshana would be upgrades only on paper because RR is already strong there.",
          "<b>Not signable:</b> Russell (retired from the IPL) and Nawaz (Pakistani player). Neesham is an IPL-untested depth option."])
st.append(P("<i>Sources for the status facts: ESPNcricinfo (Russell's retirement, 30 Nov 2025); ESPNcricinfo via RotoWire and Sportskeeda (Theekshana released by RR before the 2026 auction; unsold at the auction); Wikipedia's 2026 Rajasthan Royals season page (retentions, the Samson-for-Jadeja-and-Curran trade, coach and captain); Yahoo Sports and other outlets (retention deadline of 17 Nov 2026, lists unannounced at publication, reported RR-LSG trade talks). The match results and all performance figures come from the supplied data.</i>", "cap"))

st.append(H1("Appendix A. Parameters and design choices"))
st += [TBL([["Item", "Value", "Why / how tested"],
    ["High / normal pressure", "PI ≥ 0.75 / PI ≤ 0.50", "Top quartile; 0.70, 0.80 tested"],
    ["Chase PI blend", "0.6·leverage + 0.4·stress", "Leverage weight 0.4, 0.8 tested"],
    ["Innings-1 PI blend", "0.40 phase + 0.35 below-par + 0.25 collapse", "Judgemental; pooled and phase-neutral variants tested"],
    ["Stakes multiplier", "1 + 0.2·s (s = 0, 0.5, 1)", "Prior; 0, 0.1, 0.4 tested"],
    ["Eligibility", "≥100 pressure balls, ≥15 matches, ≥100 normal balls", "80/12 and 150/20 tested"],
    ["Component weights", "R 25, Wsave 15, Δ 20, DOT 15, BD 10, CONS 15", "Equal, value-only, no-Δ, Δ-heavy tested"],
    ["Estimator blend", "0.5 A + 0.5 B", "Flat predictive plateau 0.5-1.0"],
    ["Bootstrap", "300 match resamples, seed 11", "Rank intervals, P(top-10)"],
    ["Context models", "HistGradientBoosting, 120 iterations, 12 leaves, min leaf 400", "Low-capacity by design; features are low-cardinality"],
    ["Win-probability CV", "5 folds grouped by match (seed 7)", "Calibration, Brier, AUC reported"],
    ["Software", "Python 3.12, pandas 3.0, scikit-learn 1.9, scipy 1.18, numpy 2.5, matplotlib 3.11", "Run order in Appendix B"]], [3.6, 6.4, 7.4])]
st.append(H1("Appendix B. Reproducing the analysis"))
st += BL(["<b>prep.py</b>: audit, legal-ball rule, extras reconstruction and validation, running match state, exclusions.",
          "<b>pressure.py</b>: runs-to-go model, win probabilities with grouped CV, leverage, Pressure Index and its variants.",
          "<b>values.py</b>: context baselines, opposition effects, per-ball values.",
          "<b>rr_analysis.py / rr_tables.py / rr_fit.py / rr_gain.py</b>: Rajasthan Royals gap analysis, player profiles and indicative gains (Section 11). <b>rate.py</b>: estimators A and B, composite and bootstrap; <b>run_final.py</b> (final ratings), <b>validate_pred.py</b> / <b>validate_blend.py</b> (out-of-sample), <b>sens.py</b> (18 specifications and time split), <b>perm.py</b> (permutation test), <b>finalize.py</b> (consensus rank), <b>figs.py</b>, <b>build_report.py</b>.",
          "All files read only the six supplied CSVs. Full ranked lists with player IDs, all components, bootstrap intervals and consensus statistics are provided as CSV files with the submission."])
def appx(side, n=25):
    r = (fb if side == "bat" else fw).head(n)
    d = [["#", "Player", "ID", "Press. balls", "Matches", "Mean rank (18)", "Top-10 in # of 18"]]
    for pid, x in r.iterrows(): d.append([int(x.final_rank), x["name"], pid, int(x.nP), int(x.P_matches), f"{x.consensus_mean_rank:.1f}", int(round(x.top10_share * 18))])
    return TBL(d, [0.7, 3.4, 1.2, 1.7, 1.5, 2.3, 2.3])
st += [H2("Appendix C. Top 25 batters and bowlers (with player IDs)"), appx("bat"), Spacer(1, 8), appx("bowl")]
doc.build(st)
print("built", OUT)
