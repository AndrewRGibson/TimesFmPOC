"""Regenerates data/summary_precomputed.json -- the fixed-configuration
dataset the Summary tab loads instead of running a live sweep.

The Summary tab used to run TimesFM + every classical baseline across a
user-configurable sample on every visit, which meant a "fast" per-series
forecast (a few hundred ms) added up across dozens of series into a slow,
multi-second-to-minute wait -- and re-ran in full on every widget change in
that tab (Streamlit reruns the whole script on any interaction), not just on
an explicit "run" click. Since the underlying catalog and models only change
when someone edits this repo, there's no need to pay that cost on every page
load: this script runs the full sweep once, offline, and the app just reads
the result.

Run this whenever the synthetic catalog (src/synthetic.py) or the TimesFM
model/baseline logic changes:

    python scripts/precompute_summary.py

Uses ALL series in every category (not a sample) and every baseline model,
since cost no longer matters once this only runs offline -- the most
complete summary the catalog can produce.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import baselines  # noqa: E402
import forecasting  # noqa: E402
import synthetic  # noqa: E402
import summary_compute  # noqa: E402

HOLDOUT_FRAC = 0.15
MODEL_CHOICES = baselines.MODEL_CHOICES  # every baseline this catalog supports
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "summary_precomputed.json"


def main() -> None:
    catalog = synthetic.get_catalog()
    per_category = min(len(synthetic.series_ids_in_category(c)) for c in synthetic.list_categories())
    sample_ids = summary_compute.sample_series_ids(per_category)
    print(f"Loading TimesFM...")
    model = forecasting.load_model()

    print(f"Computing {len(sample_ids)} series x TimesFM + {len(MODEL_CHOICES)} baselines...")
    t0 = time.time()

    def progress(i: int, n: int, name: str) -> None:
        print(f"  [{i + 1}/{n}] {name}")

    rows = summary_compute.compute_summary_rows(model, sample_ids, MODEL_CHOICES, HOLDOUT_FRAC, progress)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({len(rows)} rows).")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_series": len(sample_ids),
        "n_catalog_series": len(catalog),
        "per_category": per_category,
        "model_choices": MODEL_CHOICES,
        "holdout_frac": HOLDOUT_FRAC,
        "rows": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
