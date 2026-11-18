# SupeRR Selector Season 2 · Round 1: Who delivers under pressure?

A reproducible pipeline that rates T20 batters and bowlers on how they perform **under pressure** (IPL and T20I, 2022–2026), built for Round 1 of the *SupeRR Selector Season 2* analytics challenge.

The full write-up is in [`docs/SupeRR_S2_R1_Methodology_Report.pdf`](docs/SupeRR_S2_R1_Methodology_Report.pdf). This README explains how to run the code and what each piece does.

## What it does

1. **Defines pressure for every delivery** with a Pressure Index (PI, 0–1) computed from the match state *before* the ball: calibrated win probability and its leverage for chases; below-par, collapse and end-of-innings terms for first innings; a stakes multiplier for knockouts.
2. **Scores every ball against a situational baseline**: what an average player would be expected to do in that over, wicket and competition state, adjusted for opposing players and teams, with wickets priced in runs.
3. **Rates players with empirical-Bayes shrinkage** (two estimators, blended 50/50) so small samples cannot beat proven performers.
4. **Ranks by consensus**: the final order is the mean rank across 18 pre-defined model specifications.
5. **Tests reliability**: match-clustered bootstrap, out-of-sample prediction, time-split stability, and a permutation test for clutch differentials.
6. **Translates the ratings into a Rajasthan Royals auction view** (IPL 2027): where RR is strong or weak under pressure, and how each rated player could help.

## Results

Analysed: **778 matches, 186,143 deliveries** (425 T20I + 353 IPL). Eligibility: at least 100 high-pressure balls, 15 matches and 100 normal-situation balls (111 batters, 112 bowlers rated).

| # | Batter | Bowler |
|---|---|---|
| 1 | Abhishek Sharma | Sunil Narine |
| 2 | Suryakumar Yadav | Jasprit Bumrah |
| 3 | Heinrich Klaasen | Matheesha Pathirana |
| 4 | Rajat Patidar | Varun Chakravarthy |
| 5 | Naman Dhir | Maheesh Theekshana |
| 6 | Tim David | Mohammad Nawaz |
| 7 | Phil Salt | Nathan Ellis |
| 8 | Andre Russell | Rashid Khan |
| 9 | Shreyas Iyer | Axar Patel |
| 10 | Jimmy Neesham | Josh Hazlewood |

Confidence is uneven: batters 1–6 and bowlers 1–5 are stable across specifications, while batter places 7–10 are close calls. Full ranked lists, with player IDs, all components and bootstrap intervals, are in [`results/`](results/).

Key findings (details and caveats in the report):

- The chase win-probability model reaches AUC 0.90 and Brier 0.128 out-of-fold (an RRR-only rule scores 0.172).
- Shrunk ratings predict held-out pressure performance about 1.6× (batters) and 2.0× (bowlers) better than raw pressure statistics.
- A separate "clutch gap" is not distinguishable from chance (permutation test); what is detectable is persistent quality *in* pressure situations.
- Pressure performance is only weakly persistent, so rank intervals are wide.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run_all.sh              # chains the steps below
├── data/                   # put the supplied CSVs here (not committed)
├── docs/                   # methodology report (PDF)
├── results/                # final ranked lists (CSV snapshot)
└── *.py                    # pipeline scripts, run from the repository root
```

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `prep.py` | Data audit, legal-ball rule, extras reconstruction and validation, running match state, exclusions | CSVs | `balls.pkl`, `matches_clean.pkl`, `names.pkl`, `audit.json` |
| `pressure.py` | Runs-to-go model, win probabilities (5-fold CV grouped by match), leverage, Pressure Index and its variants | `balls.pkl` | `balls_pi.pkl`, `pressure_diag.json`, `oof_chase.pkl`, `oof_inn1.pkl`, `MU.npy`, `SG.npy` |
| `values.py` | Context baselines, opposition effects, per-ball opposition-adjusted values | `balls_pi.pkl` | `bat_rows.pkl`, `bowl_rows.pkl`, `values_diag.json` |
| `rate.py` | Library: component construction, estimators A and B, composite, bootstrap | (imported) | none |
| `run_base.py` | Pressure-only estimator run; defines eligible-player sets used by validation | rows | `base_results.pkl` |
| `run_final.py` | Final blended ratings with 300-resample bootstrap | rows | `final_results.pkl` |
| `validate_pred.py`, `validate_blend.py` | Out-of-sample comparison of estimators and blend weights (100 random match splits) | rows, `base_results.pkl` | `pred_validation.pkl`, `blend_validation.pkl` |
| `sens.py` | 18 specifications, IPL-only / T20I-only runs, 2022–23 vs 2024–26 split | rows, `final_results.pkl` | `sens_results.pkl` |
| `perm.py` | Permutation test for clutch differentials | rows, `final_results.pkl` | `perm_results.json` |
| `finalize.py` | Consensus ranks (the final order) | `sens_results.pkl`, `final_results.pkl` | `fin_ranked.pkl` |
| `rr_analysis.py` | Exploratory RR-vs-league pressure summary | pickles, `teams.csv` | `rr_gap.pkl` |
| `rr_tables.py`, `rr_fit.py`, `rr_gain.py` | Rajasthan Royals section: gap table, player profiles, indicative gains | pickles, mapping CSVs | `rr_table.pkl`, `rr_profiles.pkl`, `rr_roster.pkl`, `rr_gain.pkl` |
| `export_csv.py` | Final ranked lists as CSV | `fin_ranked.pkl` | `outputs/*.csv` |
| `figs.py` | Report figures | pickles | `fig/*.png`, `effective_weights.json` |
| `build_report.py` | Assembles the PDF report | everything above | `outputs/SupeRR_S2_R1_Methodology_Report.pdf` |

## Quick start

### 1. Environment

Python 3.12. Install the versions the results were produced with:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Other recent versions will probably work but were not tested. The report uses DejaVu fonts that ship with matplotlib, so no extra fonts are needed.

### 2. Data

The raw data belongs to the challenge organisers and is **not included**. Copy the files into `data/`:

| File | Needed for |
|---|---|
| `ball_by_ball.csv`, `matches.csv`, `competitions.csv` | Core pipeline |
| `teams.csv`, `competition_team_player_mapping.csv` | Rajasthan Royals section only |
| `playing_eleven.csv` | **Not used** by this pipeline |

To use a different folder, set `DATA_DIR` (default `data`). The report and CSVs are written to `OUT_DIR` (default `outputs`).

```bash
export DATA_DIR=/path/to/csvs      # optional
export OUT_DIR=/path/to/outputs    # optional
```

### 3. Run

Everything runs from the repository root, in this order (each script reads the previous scripts' files from the working directory):

```bash
python prep.py                                   # 1  data audit and match state
python pressure.py                               # 2  Pressure Index
python values.py                                 # 3  per-ball values
python run_base.py                               # 4  eligible sets for validation
python run_final.py                              # 5  final ratings + bootstrap
python validate_pred.py && python validate_blend.py   # 6  out-of-sample validation
python sens.py                                   # 7  18 specifications + splits
python perm.py                                   # 8  permutation test
python finalize.py                               # 9  consensus ranks
python rr_analysis.py && python rr_tables.py && python rr_fit.py && python rr_gain.py   # 10 RR section
python export_csv.py                             # 11 CSV outputs
python figs.py && python build_report.py         # 12 figures and PDF
```

or `bash run_all.sh`. The pipeline was developed and run step by step on one CPU and 3 GB of RAM; as a rough estimate it takes **30–40 minutes** in total, dominated by the bootstrap (`run_final.py`) and the sensitivity grid (`sens.py`). Intermediate files are about 350 MB in total (`balls_pi.pkl` is the largest at roughly 140 MB), so they are git-ignored.

## Key parameters

Constants live at the top of `rate.py` and `pressure.py` (and are tabulated in Appendix A of the report).

| Item | Value |
|---|---|
| High / normal pressure | PI ≥ 0.75 / PI ≤ 0.50 (PI is a percentile within each innings) |
| Chase PI | 0.6 × leverage + 0.4 × stress |
| Innings-1 PI | 0.40 × end-of-innings intensity + 0.35 × below-par stress + 0.25 × collapse index |
| Stakes multiplier | 1 + 0.2·s (s = 0 league/bilateral, 0.5 Super 4, 1 knockouts) |
| Eligibility | ≥ 100 pressure balls, ≥ 15 matches, ≥ 100 normal balls |
| Component weights | scoring 25, wicket value 15, clutch gap 20, dot-ball control 15, boundaries 10, consistency 15 (%) |
| Estimator blend | 0.5 × pressure-only EB + 0.5 × hierarchical EB |
| Bootstrap | 300 match resamples |

## Reproducibility notes

- **Seeds:** win-probability CV 7 (`pressure.py`), bootstrap 11 (`run_final.py`, `rate.py`), validation splits 5, permutation test 3; gradient-boosted models use `random_state=0`. Results should match to rounding with the pinned library versions; bit-for-bit equality across library versions is not guaranteed.
- **Changing the data path** does not change results: `prep.py` reproduces the identical `audit.json` with `DATA_DIR` set.
- **Section 11 of the report** uses facts gathered from news sources on 17 November 2026 (the retention deadline day; RR's official retained and released lists were not yet published). Those facts are written into `build_report.py` as static text; the performance numbers beside them are computed. Re-check them before reusing the section.

## Data notes and assumptions

- `extras1_runs` is a categorical code, not a run count, and extras runs are not recorded. Wides and no-balls carry one extra run; byes and leg-byes are inferred from strike rotation and calibrated against 383 matches with known winning margins (mean absolute error 1.45 runs). See Section 4 of the report.
- Legal deliveries exclude wides, no-balls (`Overstep`) and the `other` no-ball variants.
- Excluded: the two Emerging Teams Asia Cups (30 matches), matches with a revised target or D/L text, no result or no ball data, curtailed innings, and super-over balls. The IPL contributes 5 seasons (2022–26), not 7.
- Player IDs are the key; two names (Abdul Samad, Haider Ali) map to two IDs each and are disambiguated by ID.

## Limitations

Pressure is a model, not an observation; the first-innings win-probability model is only moderately informative (AUC 0.70), and its stress weights are judgemental (tested by sensitivity). Pressure samples are small (about 100–600 balls per player), so ranks beyond the top few are uncertain. No pitch, weather or fielding data are available. The stakes multiplier is a prior, not an estimate. See Section 10 of the report.

## Licence and data

Add a `LICENSE` file for the code before publishing. The challenge data is not redistributed here and remains subject to the organisers' terms.

## Acknowledgements

Built for SupeRR Selector Season 2 (Rajasthan Royals × Unstop).
