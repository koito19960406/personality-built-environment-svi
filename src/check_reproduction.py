"""Compare freshly reproduced model results against the published outputs.

Reads the tidy coefficient tables from ``results/reproduced/`` and
``results/published_model_outputs/``, joins them on (trait, level, term) and
reports the largest absolute discrepancy in the coefficients, standard errors
and p-values.

Usage:
    python src/reproduce_models.py      # first, to produce results/reproduced/
    python src/check_reproduction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO = PROJECT_ROOT / "results" / "reproduced"
PUBLISHED = PROJECT_ROOT / "results" / "published_model_outputs"

# Coefficients are on a z-scored scale, so an absolute tolerance is appropriate.
# OLS is a closed-form solve and matches to machine precision. The spatial lag
# models are fit by maximum likelihood, so their coefficients move slightly with
# the SciPy/spreg version used; 1e-4 is well below the third decimal place the
# published tables report.
TOL_OLS = 1e-6
TOL_SAR = 1e-4


def compare(name: str, filename: str, key: list[str], value_cols: list[str], tol: float) -> bool:
    a = pd.read_csv(REPRO / filename)
    b = pd.read_csv(PUBLISHED / filename)

    merged = a.merge(b, on=key, how="outer", suffixes=("_repro", "_pub"), indicator=True)
    only = merged[merged["_merge"] != "both"]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"  rows: reproduced={len(a)}  published={len(b)}  matched={int((merged['_merge'] == 'both').sum())}")

    ok = True
    if len(only):
        ok = False
        print(f"  UNMATCHED ROWS: {len(only)}")
        print(only[key + ["_merge"]].head(10).to_string(index=False))

    both = merged[merged["_merge"] == "both"]
    for col in value_cols:
        lhs, rhs = f"{col}_repro", f"{col}_pub"
        if lhs not in both.columns:
            continue

        left, right = both[lhs], both[rhs]
        # A value missing in both files is a match, not a discrepancy. The
        # spatial lag tidy tables carry no standard errors in either file
        # (see the note in the README); the summary .txt files do.
        both_nan = left.isna() & right.isna()
        one_nan = int((left.isna() ^ right.isna()).sum())
        if one_nan:
            ok = False
            print(f"  FAIL {col}: {one_nan} rows missing on exactly one side")

        diff = (left - right).abs()[~both_nan].dropna()
        if diff.empty:
            print(f"  --   {col}: no comparable values (missing in both files)")
            continue

        worst = float(diff.max())
        if worst > tol:
            ok = False
        print(f"  {'OK ' if worst <= tol else 'FAIL'} max |Δ {col}| = {worst:.3e}")

    return ok


def main() -> int:
    checks = [
        compare(
            "OLS coefficients",
            "ols_results_tidy_all.csv",
            key=["trait", "level", "term"],
            value_cols=["coef", "se", "t", "p", "n", "r2", "adj_r2"],
            tol=TOL_OLS,
        ),
        compare(
            "Spatial lag coefficients",
            "sar_results_tidy_all.csv",
            key=["trait", "level", "variable"],
            value_cols=["coefficient", "se", "z", "p", "n", "logll", "moran_I_resid"],
            tol=TOL_SAR,
        ),
    ]

    print()
    if all(checks):
        print(
            "PASS — reproduced results match the published outputs "
            f"(OLS tol = {TOL_OLS:g}, SAR tol = {TOL_SAR:g})."
        )
        return 0
    print("FAIL — see the discrepancies above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
