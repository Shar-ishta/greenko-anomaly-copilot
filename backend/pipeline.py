"""Anomaly detection pipeline for renewable-asset telemetry.

Stages: load/validate -> physics-informed baseline -> robust z-score features
-> statistical rules + IsolationForest -> ensemble/severity -> events
-> playbook retrieval (TF-IDF) -> operator-language explanation.
Pure pandas/numpy/scikit-learn so it runs offline.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "asset_telemetry.csv"
KB_PATH = BASE / "data" / "knowledge_base.json"

REQUIRED = ["timestamp_id", "asset_id", "hour", "irradiance_wm2", "wind_speed_ms",
            "temperature_c", "vibration_mm_s", "power_mw"]

CONFIG = dict(
    z_thresh=3.5,            # Iglewicz-Hoaglin modified z-score cut-off
    scale_floor_frac=0.03,   # min power-residual scale = 3% of rated power
    vib_floor=0.10,          # mm/s, min scale for vibration z
    temp_floor=1.5,          # deg C, min scale for temperature z
    if_contamination=0.03,   # IsolationForest expected anomaly share
    if_min_z=2.0,            # IF flag only counts if some signal is also >= this
    n_estimators=300,
    random_state=42,
    min_perf_frac=0.10,      # underperformance rule only when expected >= 10% rated
    event_gap=2,             # merge flagged points up to 2 steps apart
)

RULE_TEXT = {
    "HIGH_VIBRATION": "vibration far above this asset's normal level",
    "OVER_TEMPERATURE": "temperature far above normal for this hour of day",
    "UNDERPERFORMANCE": "power well below what weather conditions should produce",
    "OVER_REPORTING": "power well above what weather conditions should produce",
    "NIGHT_GENERATION": "power reported while there is no sunlight",
    "SENSOR_FLATLINE": "sensor value frozen (no variation)",
    "DATA_QUALITY": "missing or physically impossible reading",
}


def asset_type(asset_id: str) -> str:
    a = str(asset_id).upper()
    return "solar" if a.startswith("SOLAR") else "wind" if a.startswith(("WTG", "WIND")) else "unknown"


# ------------------------------------------------------------------ loading
def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path) if not isinstance(path, pd.DataFrame) else path.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in REQUIRED[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["asset_id", "timestamp_id"]).reset_index(drop=True)
    return df


def _dq(g: pd.DataFrame) -> pd.Series:
    bad = g[["irradiance_wm2", "wind_speed_ms", "temperature_c", "vibration_mm_s", "power_mw"]].isna().any(axis=1)
    bad |= (g.power_mw < 0) | (g.vibration_mm_s < 0) | (g.irradiance_wm2 < 0) | (g.irradiance_wm2 > 1500)
    bad |= ~g.hour.between(0, 23) | (g.temperature_c.abs() > 120)
    return bad


# ----------------------------------------------------------------- features
def robust_z(x: pd.Series, floor: float) -> pd.Series:
    med = x.median()
    scale = max(1.4826 * (x - med).abs().median(), floor)
    return (x - med) / scale


def _fit_baseline(g: pd.DataFrame, kind: str, rated: float, floor: float) -> np.ndarray:
    """Expected power from weather, fitted robustly so faults do not bias it."""
    p, ok = g.power_mw.values, ~g.power_mw.isna().values
    if kind == "solar":
        x = g.irradiance_wm2.values
        keep = ok & (x > 50)
        for _ in range(3):
            if keep.sum() < 3:
                break
            k = np.median(p[keep] / x[keep])
            r = p - k * x
            keep = ok & (x > 50) & (np.abs(r) <= 3 * max(1.4826 * np.median(np.abs(r[keep])), floor))
        k = np.median(p[ok & (x > 50)] / x[ok & (x > 50)]) if (ok & (x > 50)).any() else 0.0
        return np.clip(k * np.nan_to_num(x), 0, None)
    x = g.wind_speed_ms.values
    keep = ok & ~np.isnan(x)
    coef = np.polyfit(x[keep], p[keep], 3)
    for _ in range(4):  # iterative trimming
        r = p - np.polyval(coef, np.nan_to_num(x))
        s = max(1.4826 * np.median(np.abs(r[keep])), floor)
        keep = ok & ~np.isnan(x) & (np.abs(r) <= 2.5 * s)
        if keep.sum() < 8:
            break
        coef = np.polyfit(x[keep], p[keep], 3)
    return np.clip(np.polyval(coef, np.nan_to_num(x)), 0, None)


def build_features(df: pd.DataFrame, cfg=CONFIG) -> pd.DataFrame:
    out = []
    for aid, g in df.groupby("asset_id", sort=False):
        g = g.copy().reset_index(drop=True)
        kind = asset_type(aid)
        rated = float(np.nanpercentile(g.power_mw, 99)) or 1.0
        floor = cfg["scale_floor_frac"] * rated
        g["asset_type"], g["rated_mw"] = kind, rated
        g["step"] = np.arange(len(g))
        g["dq_issue"] = _dq(g)
        g[["irradiance_wm2", "wind_speed_ms", "temperature_c", "vibration_mm_s", "power_mw"]] = \
            g[["irradiance_wm2", "wind_speed_ms", "temperature_c", "vibration_mm_s", "power_mw"]].interpolate(limit_direction="both")
        g["expected_mw"] = _fit_baseline(g, kind, rated, floor)
        g["power_resid"] = g.power_mw - g.expected_mw
        g["power_z"] = robust_z(g.power_resid, floor)
        g["vib_baseline"] = g.vibration_mm_s.median()
        g["vib_z"] = robust_z(g.vibration_mm_s, cfg["vib_floor"])
        prof = g.groupby("hour").temperature_c.transform("median")  # diurnal profile
        g["temp_baseline"] = prof
        g["temp_resid"] = g.temperature_c - prof
        g["temp_z"] = robust_z(g.temp_resid, cfg["temp_floor"])
        g["vib_roll3"] = g.vib_z.rolling(3, min_periods=1).mean()
        g["d_temp"] = g.temperature_c.diff().fillna(0)
        ratio = np.where(g.expected_mw >= cfg["min_perf_frac"] * rated, g.power_mw / g.expected_mw.clip(lower=1e-6), 1.0)
        g["power_ratio"] = np.clip(ratio, 0, 2)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- detection
def apply_rules(f: pd.DataFrame, cfg=CONFIG) -> pd.DataFrame:
    zt = cfg["z_thresh"]
    f = f.copy()
    active = f.expected_mw >= cfg["min_perf_frac"] * f.rated_mw
    flat = pd.Series(False, index=f.index)
    for aid, g in f.groupby("asset_id"):
        for col in ("vibration_mm_s", "temperature_c"):
            flat.loc[g.index] |= (g[col].rolling(6).std() < 1e-9).fillna(False)
    night = (f.asset_type == "solar") & (f.irradiance_wm2 < 5) & (f.power_mw > 0.02 * f.rated_mw)
    f["r_HIGH_VIBRATION"] = f.vib_z >= zt
    f["r_OVER_TEMPERATURE"] = f.temp_z >= zt
    f["r_UNDERPERFORMANCE"] = (f.power_z <= -zt) & active
    f["r_OVER_REPORTING"] = (f.power_z >= zt) & active
    f["r_NIGHT_GENERATION"] = night
    f["r_SENSOR_FLATLINE"] = flat
    f["r_DATA_QUALITY"] = f.dq_issue
    f["rules"] = f.apply(lambda r: [k[2:] for k in RULE_TEXT_KEYS if r[k]], axis=1)
    f["n_rules"] = f.rules.str.len()
    return f


RULE_TEXT_KEYS = ["r_" + k for k in RULE_TEXT]
IF_FEATURES = ["power_z", "vib_z", "temp_z", "vib_roll3", "d_temp", "power_ratio"]


def apply_isolation_forest(f: pd.DataFrame, cfg=CONFIG) -> pd.DataFrame:
    f = f.copy()
    f["if_flag"], f["if_rank"] = False, 0.0
    for aid, g in f.groupby("asset_id"):
        X = g[IF_FEATURES].clip(-15, 15).fillna(0).values
        m = IsolationForest(n_estimators=cfg["n_estimators"], contamination=cfg["if_contamination"],
                            random_state=cfg["random_state"]).fit(X)
        s = -m.score_samples(X)                       # higher = more anomalous
        rank = pd.Series(s, index=g.index).rank(pct=True)
        max_z = g[["power_z", "vib_z", "temp_z"]].abs().max(axis=1)
        f.loc[g.index, "if_rank"] = rank
        f.loc[g.index, "if_flag"] = (m.predict(X) == -1) & (max_z >= cfg["if_min_z"]).values
    return f


def score_and_flag(f: pd.DataFrame, use_rules=True, use_if=True) -> pd.DataFrame:
    f = f.copy()
    max_z = f[["power_z", "vib_z", "temp_z"]].abs().max(axis=1)
    rule_hit = (f.n_rules > 0) if use_rules else pd.Series(False, index=f.index)
    if_hit = f.if_flag if use_if else pd.Series(False, index=f.index)
    f["is_anomaly"] = rule_hit | if_hit
    f["anomaly_score"] = (100 * (0.6 * (max_z / 10).clip(upper=1) + 0.4 * f.if_rank)).round(1)
    sev = np.where(f.is_anomaly, "low", "normal")
    sev = np.where(rule_hit, "medium", sev)
    sev = np.where((f.n_rules >= 2) | ((max_z >= 6) & rule_hit), "high", sev)
    f["severity"] = sev
    return f


def run_pipeline(df: pd.DataFrame | None = None, cfg=CONFIG, use_rules=True, use_if=True) -> pd.DataFrame:
    df = load_data() if df is None else df
    f = apply_isolation_forest(apply_rules(build_features(df, cfg), cfg), cfg)
    return score_and_flag(f, use_rules, use_if)


# ------------------------------------------------------------------- events
_SEV = {"normal": 0, "low": 1, "medium": 2, "high": 3}


def extract_events(f: pd.DataFrame, cfg=CONFIG) -> list[dict]:
    events = []
    for aid, g in f.groupby("asset_id", sort=False):
        idx = g.index[g.is_anomaly].tolist()
        groups, cur = [], []
        for i in idx:
            if cur and g.loc[i, "step"] - g.loc[cur[-1], "step"] > cfg["event_gap"]:
                groups.append(cur); cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        for n, ids in enumerate(groups, 1):
            e = f.loc[ids]
            rules = sorted({r for rs in e.rules for r in rs})
            shortfall = float((e.expected_mw - e.power_mw).clip(lower=0).sum())
            ev = dict(
                event_id=f"{aid}-E{n}", asset_id=aid, asset_type=e.asset_type.iloc[0],
                start_id=int(e.timestamp_id.min()), end_id=int(e.timestamp_id.max()),
                day=int(e.step.iloc[0] // 24) + 1, start_hour=int(e.hour.iloc[0]), end_hour=int(e.hour.iloc[-1]),
                n_points=len(e), rules=rules,
                severity=max(e.severity, key=_SEV.get),
                peak_score=float(e.anomaly_score.max()),
                peak_vibration=float(e.vibration_mm_s.max()), normal_vibration=float(e.vib_baseline.iloc[0]),
                peak_vib_z=float(e.vib_z.max()),
                peak_temp=float(e.temperature_c.max()), temp_above_normal=float(e.temp_resid.max()),
                peak_temp_z=float(e.temp_z.max()),
                mean_power=float(e.power_mw.mean()), mean_expected=float(e.expected_mw.mean()),
                energy_shortfall_mwh=round(shortfall, 3), min_power_z=float(e.power_z.min()),
                if_agrees=bool(e.if_flag.any()),
            )
            ev["explanation"] = explain_event(ev)
            events.append(ev)
    return sorted(events, key=lambda x: (-_SEV[x["severity"]], -x["peak_score"]))


# ---------------------------------------------------- retrieval + explanation
def _load_kb():
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def _signature(ev: dict) -> str:
    words = {"HIGH_VIBRATION": "high vibration", "OVER_TEMPERATURE": "high temperature overheating",
             "UNDERPERFORMANCE": "low power output below expected underperformance",
             "OVER_REPORTING": "power above expected over reporting",
             "NIGHT_GENERATION": "power at night no irradiance", "SENSOR_FLATLINE": "flatline stuck frozen sensor",
             "DATA_QUALITY": "missing data impossible value"}
    s = " ".join(words[r] for r in ev["rules"]) + f" {ev['asset_type']}"
    if ev["asset_type"] == "solar" and ev["mean_expected"] < 0.05:
        s += " no generation night standby"
    if not ev["rules"]:
        s += " subtle statistical outlier"
    return s


def retrieve_playbooks(ev: dict, k: int = 2) -> list[dict]:
    kb = [e for e in _load_kb() if ev["asset_type"] in e["asset_types"] or "any" in e["asset_types"]]
    docs = [e["title"] + " " + e["symptoms"] for e in kb]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit(docs + [_signature(ev)])
    sims = cosine_similarity(vec.transform([_signature(ev)]), vec.transform(docs))[0]
    top = np.argsort(-sims)[:k]
    return [dict(kb[i], similarity=round(float(sims[i]), 3)) for i in top if sims[i] > 0]


def explain_event(ev: dict) -> dict:
    when = f"Day {ev['day']}, {ev['start_hour']:02d}:00" + (f"–{ev['end_hour']:02d}:00" if ev["end_hour"] != ev["start_hour"] else "")
    facts = []
    if "HIGH_VIBRATION" in ev["rules"]:
        facts.append(f"vibration peaked at {ev['peak_vibration']:.2f} mm/s (normal ≈ {ev['normal_vibration']:.2f}, about {ev['peak_vib_z']:.0f}× the usual spread)")
    if "OVER_TEMPERATURE" in ev["rules"]:
        facts.append(f"temperature reached {ev['peak_temp']:.1f} °C, {ev['temp_above_normal']:.1f} °C above normal for that hour")
    if "UNDERPERFORMANCE" in ev["rules"]:
        facts.append(f"output averaged {ev['mean_power']:.2f} MW against ≈ {ev['mean_expected']:.2f} MW expected for the weather (≈ {ev['energy_shortfall_mwh']:.2f} MWh lost)")
    if "OVER_REPORTING" in ev["rules"]:
        facts.append(f"output ({ev['mean_power']:.2f} MW) was above the ≈ {ev['mean_expected']:.2f} MW the weather supports")
    if "NIGHT_GENERATION" in ev["rules"]:
        facts.append("power was reported with no sunlight")
    if "SENSOR_FLATLINE" in ev["rules"]:
        facts.append("a sensor stopped changing")
    if "DATA_QUALITY" in ev["rules"]:
        facts.append("a reading was missing or physically impossible")
    if not facts:
        facts.append("the combination of readings is unusual compared with the rest of this asset's history, although no single signal crossed a hard limit")
    standby = ev["asset_type"] == "solar" and ev["mean_expected"] < 0.05
    plays = retrieve_playbooks(ev)
    note = ("No generation is lost because the plant was not producing at the time, but heat and vibration while idle point to a "
            "mechanical or cooling problem that will cost output once the sun is up." if standby and "HIGH_VIBRATION" in ev["rules"] else "")
    summary = f"{ev['asset_id']} · {when}: " + "; ".join(facts) + "." + (" " + note if note else "")
    confidence = "high" if (len(ev["rules"]) >= 2 and ev["if_agrees"]) else "medium" if ev["rules"] else "low"
    out = dict(
        summary=summary, when=when, confidence=confidence,
        why_flagged=[RULE_TEXT[r] for r in ev["rules"]] + (["IsolationForest also found this combination unusual"] if ev["if_agrees"] else []),
        likely_causes=[c for p in plays for c in p["causes"]][:5],
        next_checks=[c for p in plays for c in p["checks"]][:6],
        safety_note=plays[0]["safety"] if plays else "Follow site LOTO and OEM procedures before inspecting equipment.",
        playbooks=[{"id": p["id"], "title": p["title"], "similarity": p["similarity"]} for p in plays],
        generated_by="rules+retrieval",
    )
    return maybe_llm_polish(out)


def maybe_llm_polish(expl: dict) -> dict:
    """Optional: rewrite the summary in plainer language via an LLM. Facts stay grounded in the
    computed evidence; any failure silently falls back to the deterministic text.
    Enable with ANTHROPIC_API_KEY (never hard-coded)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or os.environ.get("USE_LLM", "1") == "0":
        return expl
    try:
        body = json.dumps({
            "model": os.environ.get("LLM_MODEL", "claude-sonnet-4-6"), "max_tokens": 300,
            "messages": [{"role": "user", "content":
                "Rewrite for a plant operator in 2-3 plain sentences. Use ONLY these facts, add nothing:\n" + expl["summary"]}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            txt = json.loads(r.read())["content"][0]["text"].strip()
        return dict(expl, summary_llm=txt, generated_by="rules+retrieval+llm")
    except Exception:
        return expl


# ----------------------------------------------------------------- summary
def summarize(f: pd.DataFrame, events: list[dict]) -> dict:
    return dict(
        n_rows=int(len(f)), n_assets=int(f.asset_id.nunique()), n_flagged_points=int(f.is_anomaly.sum()),
        n_events=len(events), by_severity={s: sum(e["severity"] == s for e in events) for s in ("high", "medium", "low")},
        total_energy_shortfall_mwh=round(sum(e["energy_shortfall_mwh"] for e in events), 3),
    )


def series_payload(f: pd.DataFrame, asset_id: str) -> list[dict]:
    cols = ["timestamp_id", "step", "hour", "irradiance_wm2", "wind_speed_ms", "temperature_c", "vibration_mm_s",
            "power_mw", "expected_mw", "temp_baseline", "vib_z", "temp_z", "power_z", "anomaly_score", "severity", "is_anomaly"]
    g = f[f.asset_id == asset_id][cols].round(3)
    return json.loads(g.to_json(orient="records"))
