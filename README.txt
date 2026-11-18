SupeRR Selector S2 - Round 1: reproducible pipeline
Inputs : the six supplied CSVs (paths at the top of prep.py: /mnt/user-data/uploads/).
Run order (single CPU, roughly 30-40 minutes in total):
  1. python prep.py              -> balls.pkl, matches_clean.pkl, names.pkl, audit.json
  2. python pressure.py          -> balls_pi.pkl (Pressure Index and variants), pressure_diag.json, out-of-fold win-prob files
  3. python values.py            -> bat_rows.pkl, bowl_rows.pkl (per-ball, opposition-adjusted values)
  4. python run_base.py          -> base_results.pkl (eligible-player sets used by the validation scripts)
  5. python run_final.py         -> final_results.pkl (blended ratings + 300-resample bootstrap)
  6. python validate_pred.py ; python validate_blend.py   -> out-of-sample validation of the estimators
  7. python sens.py              -> sens_results.pkl (18 specifications, IPL-only / T20I-only, time split)
  8. python perm.py              -> perm_results.json (permutation test for clutch differentials)
  9. python finalize.py          -> fin_ranked.pkl (consensus ranks = final order)
 10. python rr_analysis.py ; python rr_tables.py ; python rr_fit.py ; python rr_gain.py   -> Rajasthan Royals section (Section 11)
 11. python figs.py ; python build_report.py   -> figures and the PDF report
Seeds: win-probability CV 7, bootstrap 11, validation splits 5, permutation test 3.
Gates, weights and thresholds are constants at the top of rate.py and pressure.py (Appendix A of the report).
