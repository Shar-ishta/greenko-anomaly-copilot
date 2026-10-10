"""Evaluation: (1) sanity check on bundled data, (2) synthetic fault-injection benchmark.

The bundled CSV has no labels, so (2) is the quantitative evidence: known faults are injected
into a clean copy of the data (bundled fault rows removed) and detection is measured.
Run:  python -m backend.evaluate   -> writes evaluation_results.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import CONFIG, DATA_PATH, load_data, run_pipeline, extract_events

OUT = Path(__file__).resolve().parents[1] / "evaluation_results.json"
LEVELS = {  # (vib multiplier, temp rise degC, power multiplier)
    "obvious": (3.0, 18, 0.45), "moderate": (2.2, 11, 0.70), "subtle": (1.6, 6, 0.88)}
KINDS = ["vibration", "overtemp", "underperf", "bearing_combo"]


def clean_series() -> pd.DataFrame:
    df = load_data()
    flagged = run_pipeline(df)
    bad = flagged.loc[flagged.is_anomaly, ["asset_id", "timestamp_id"]]
    keep = ~df.set_index(["asset_id", "timestamp_id"]).index.isin(bad.set_index(["asset_id", "timestamp_id"]).index)
    return df[keep].reset_index(drop=True)


def inject(df, rng, level):
    df = df.copy()
    vm, td, pm = LEVELS[level]
    truth = pd.Series(False, index=df.index)
    for aid, g in df.groupby("asset_id"):
        starts = []
        while len(starts) < 3:
            s = int(rng.integers(6, len(g) - 6))
            if all(abs(s - t) > 8 for t in starts):
                starts.append(s)
        for s, kind in zip(starts, rng.choice(KINDS, 3, replace=False)):
            ids = g.index[s:s + int(rng.integers(1, 4))]
            if kind in ("vibration", "bearing_combo"):
                df.loc[ids, "vibration_mm_s"] *= vm
            if kind in ("overtemp", "bearing_combo"):
                df.loc[ids, "temperature_c"] += td
            if kind in ("underperf", "bearing_combo"):
                df.loc[ids, "power_mw"] *= pm
            # an underperf on a zero-power (night) point is not observable -> don't count as truth
            observable = df.loc[ids, "power_mw"] > 0.05 if kind == "underperf" else pd.Series(True, index=ids)
            truth.loc[ids[observable.values]] = True
    return df, truth


def score(pred, truth, f):
    tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    # event-level recall: contiguous true runs with >=1 flagged point
    hit = tot = 0
    for aid, g in f.groupby("asset_id"):
        t = truth.loc[g.index].values; p_ = pred.loc[g.index].values
        i = 0
        while i < len(t):
            if t[i]:
                j = i
                while j + 1 < len(t) and t[j + 1]:
                    j += 1
                tot += 1; hit += int(p_[i:j + 1].any()); i = j + 1
            else:
                i += 1
    return tp, fp, fn, hit, tot


def benchmark(trials=30):
    base, res = clean_series(), {}
    for level in LEVELS:
        for name, (ur, ui) in {"rules_only": (True, False), "isolation_forest_only": (False, True), "ensemble": (True, True)}.items():
            agg = dict(tp=0, fp=0, fn=0, hit=0, tot=0, hours=0)
            for t in range(trials):
                df, truth = inject(base, np.random.default_rng(1000 + t), level)
                f = run_pipeline(df, use_rules=ur, use_if=ui)
                tp, fp, fn, hit, tot = score(f.is_anomaly, _align(f, df, truth), f)
                for k, v in zip(("tp", "fp", "fn", "hit", "tot"), (tp, fp, fn, hit, tot)):
                    agg[k] += v
                agg["hours"] += len(f)
            p = agg["tp"] / max(agg["tp"] + agg["fp"], 1); r = agg["tp"] / max(agg["tp"] + agg["fn"], 1)
            res.setdefault(level, {})[name] = dict(
                precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / max(p + r, 1e-9), 3),
                event_recall=round(agg["hit"] / max(agg["tot"], 1), 3),
                false_alarms_per_100_hours=round(100 * agg["fp"] / agg["hours"], 2))
    return res


def _align(f, df, truth):
    """f is sorted by asset/timestamp like df (load_data sorts and resets index) -> same row order."""
    return pd.Series(truth.values, index=f.index)


def bundled_sanity():
    f = run_pipeline(); ev = extract_events(f)
    # Reference windows identified by manual inspection (EDA), NOT supplied labels:
    ref = {("SOLAR_INV_01", 67, 69), ("WTG_07", 173, 174)}
    got = {(e["asset_id"], e["start_id"], e["end_id"]) for e in ev}
    return dict(reference_windows=sorted(ref), detected_events=sorted(got), exact_match=ref == got,
                flagged_points=int(f.is_anomaly.sum()), total_points=int(len(f)))


if __name__ == "__main__":
    out = dict(config=CONFIG, bundled_data_sanity_check=bundled_sanity(), injection_benchmark=benchmark(),
               notes=["Injection benchmark: 30 random trials per level, 3 faults per asset per trial, 2 assets, seeds 1000-1029.",
                      "Recall counts only observable faults (underperformance at zero-power hours is excluded).",
                      "Synthetic faults are simpler than real ones; treat as a relative comparison and lower bound on difficulty, not a field accuracy claim."])
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["bundled_data_sanity_check"], indent=1, default=str))
    for lv, d in out["injection_benchmark"].items():
        print(lv)
        for m, v in d.items():
            print("  ", m, v)
