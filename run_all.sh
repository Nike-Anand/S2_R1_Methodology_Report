#!/usr/bin/env bash
# Chains the full pipeline. Run from the repository root. See README.md for details.
set -euo pipefail
export DATA_DIR="${DATA_DIR:-data}"
export OUT_DIR="${OUT_DIR:-outputs}"

python prep.py
python pressure.py
python values.py
python run_base.py
python run_final.py
python validate_pred.py
python validate_blend.py
python sens.py
python perm.py
python finalize.py
python rr_analysis.py
python rr_tables.py
python rr_fit.py
python rr_gain.py
python export_csv.py
python figs.py
python build_report.py
echo "Done. Report and CSVs are in $OUT_DIR/"
