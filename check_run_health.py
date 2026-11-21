"""Report whether a SCULPTOR run directory actually evaluated its candidates.

Usage:
    python check_run_health.py <run_dir> [<run_dir> ...]

The check that matters is the ratio between the candidates that reached the
property stage and the candidates that came out of it.  A run that reports
"many candidates, almost no Tier 1" is usually not a screening result at all:
the property stage failed for every molecule, and each failure was written as
an individual candidate failure instead of stopping the run.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def summarise(directory: Path) -> None:
    print(f"=== {directory}")
    summaries = sorted(directory.rglob("summary.json"))
    summaries += sorted(directory.rglob("perturbation_pipeline_summary.json"))
    if not summaries:
        print("    no summary.json found")
        return

    reached = classified = failed = 0
    for path in summaries:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = data.get("property_prediction_rows")
        done = data.get("property_classified_rows")
        bad = data.get("property_failed_count")
        if rows is None and done is None and bad is None:
            continue
        reached += rows or 0
        classified += done or 0
        failed += bad or 0
        print(f"    {path.parent.name:>10s}  reached={rows}  classified={done}"
              f"  failed={bad}  tier1={data.get('tier1_count')}")

    print(f"    total      reached={reached}  classified={classified}"
          f"  failed={failed}")
    if reached and classified == 0:
        print("    VERDICT: the property stage failed for every candidate - "
              "this run produced no screening result.")
    elif failed and failed >= 0.5 * max(reached, 1):
        print("    VERDICT: more than half of the property evaluations "
              "failed; check the failure reasons below.")
    else:
        print("    VERDICT: property evaluation ran normally.")

    reasons = Counter()
    for path in directory.rglob("property_failed_candidates.csv"):
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                reason = (row.get("property_failure_reason")
                          or row.get("failure_reasons")
                          or "").strip()
                reasons[reason or "(no reason recorded)"] += 1
    for path in directory.rglob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in ("No module named", "Traceback"):
            if marker in text:
                reasons[f"{marker} (in {path.name})"] += text.count(marker)
    if reasons:
        print("    failure signatures:")
        for reason, count in reasons.most_common(5):
            print(f"      {count:5d}  {reason[:110]}")
    print()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for argument in sys.argv[1:]:
        directory = Path(argument)
        if not directory.is_dir():
            print(f"=== {directory}\n    not a directory\n")
            continue
        summarise(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
