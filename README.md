# Renewable Asset Fault & Anomaly Detection Copilot

Detects abnormal behaviour in solar-inverter and wind-turbine telemetry and explains it to operators in plain language: what happened, how bad it is, likely causes, and what to check next.

**Live URL (Railway):** https://web-production-278fd.up.railway.app/

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload          # UI at http://localhost:8000
python -m backend.evaluate                # regenerate evaluation_results.json
pip install -r requirements-dev.txt && pytest -q
```

## Railway deployment
1. Push this folder to a GitHub repo (or `railway up` from the CLI).
2. Railway → New Project → Deploy from repo. `Procfile`/`railway.json` start `uvicorn` on `$PORT`; `.python-version` pins Python 3.11; health check is `/api/health`.
3. Settings → Networking → *Generate Domain*, then paste the URL above.
4. Optional: set `ANTHROPIC_API_KEY` as a Railway variable to enable LLM rewording of summaries (off by default; no key is stored in code).

## Architecture
```
CSV -> load/validate -> per-asset baseline -> robust z-score features
    -> [Rules] + [IsolationForest] -> ensemble + severity -> events
    -> TF-IDF retrieval over troubleshooting playbooks -> operator explanation
    -> FastAPI (/api/*) -> single-page HTML dashboard (SVG charts)
```
| Path | Purpose |
|---|---|
| `backend/pipeline.py` | features, rules, IsolationForest, events, retrieval, explanations |
| `backend/app.py` | API: `/api/demo`, `/api/analyze`, `/api/analyze_upload`, `/api/run` (copilot Q&A), `/api/evaluation`, `/api/health` |
| `backend/evaluate.py` | sanity check + synthetic fault-injection benchmark |
| `data/knowledge_base.json` | 11 troubleshooting playbooks (causes, checks, safety) |
| `frontend/index.html` | dashboard: charts, events, copilot box, upload, evaluation table |

## Design choices and model logic
- **Physics-informed baseline instead of raw thresholds.** Power is judged against what the weather supports: solar = linear in irradiance; wind = cubic fit on wind speed. Both are fitted robustly (iterative trimming) so faults don't contaminate the baseline. Temperature is compared to the asset's own median for that hour of day, since temperature has a daily cycle.
- **Robust z-scores** (median/MAD, with floors so a near-zero MAD can't explode the score). Rules fire at |z| ≥ 3.5, the standard modified-z cut-off.
- **Rules (interpretable):** high vibration, over-temperature, underperformance, over-reporting, power at night, sensor flatline, data-quality (missing/impossible values).
- **IsolationForest (per asset)** on 6 engineered features (power/vibration/temperature z-scores, 3-step rolling vibration, temperature delta, power ratio). It catches unusual *combinations*. An IF flag only counts if some signal is also ≥ 2σ, which suppresses edge-of-distribution noise.
- **Ensemble:** anomaly = rule OR IF. Severity: *high* = ≥ 2 rules or z ≥ 6; *medium* = one rule; *low* = IF only. Consecutive points merge into events.
- **Explanations (GenAI step, grounded):** each event builds a signature, retrieves the best-matching playbooks by TF-IDF cosine similarity, and fills a template with the computed numbers, ranked causes, checks and a safety note. An optional LLM can reword the text, restricted to the computed facts; it falls back to the template on any error.
- **Copilot Q&A (`/api/run`):** deterministic intent routing (asset, type, severity, "what to check"). Intentionally not a free-form LLM chat.

## Results on the bundled data
Two events, 5 of 240 readings flagged, no other alerts:
| Event | When | Signals | Severity |
|---|---|---|---|
| SOLAR_INV_01 | day 3, 19:00–21:00 | vibration ≈ 3.9 mm/s (normal ≈ 1.4), temp up to 53.5 °C (+19 °C vs. normal for the hour); no generation (night) | high |
| WTG_07 | day 3, 05:00–06:00 | vibration ≈ 3.8 mm/s, temp +19 °C, output ≈ 0.72 MW vs ≈ 1.30 MW expected (≈ 1.17 MWh lost) | high |

These match the two fault windows I identified by manual inspection of the data. **That is a sanity check, not a performance claim:** the bundled CSV has no labels.

## Evaluation (synthetic fault injection)
Because there are no labels, I removed the detected fault rows to make a clean series, injected known faults (vibration spike, over-temperature, underperformance, combined bearing fault; 1–3 hours long; 3 per asset per trial) at three severities, and ran 30 seeded trials per level (`python -m backend.evaluate`).

| Fault level | Method | Precision | Recall | F1 | Event recall | False alarms /100 h |
|---|---|---|---|---|---|---|
| obvious | rules | 1.00 | 0.975 | 0.987 | 0.988 | 0.0 |
| obvious | IsolationForest | 1.00 | 0.602 | 0.752 | 0.808 | 0.0 |
| obvious | **ensemble** | 1.00 | 0.986 | 0.993 | 0.994 | 0.0 |
| moderate | rules | 1.00 | 0.942 | 0.970 | 0.946 | 0.0 |
| moderate | IsolationForest | 1.00 | 0.594 | 0.745 | 0.790 | 0.0 |
| moderate | **ensemble** | 1.00 | 0.967 | 0.983 | 0.988 | 0.0 |
| subtle | rules | 1.00 | 0.680 | 0.809 | 0.778 | 0.0 |
| subtle | IsolationForest | 1.00 | 0.506 | 0.672 | 0.653 | 0.0 |
| subtle | **ensemble** | 1.00 | 0.743 | 0.853 | 0.850 | 0.0 |

Takeaways: the ensemble beats either method alone; IsolationForest alone is weaker on single-signal faults but adds recall on top of the rules; subtle faults (≈ +60 % vibration, +6 °C, −12 % power) are where it misses most.

## Assumptions
- Hourly readings; 5 days per asset; one asset per type; schema as in the bundled CSV.
- The bundled data are mostly healthy, so median/MAD describe "normal". Ratings use the 99th-percentile power.
- Expected power depends only on irradiance (solar) or wind speed (wind).

## Limitations and failure cases
- **Small and synthetic data** (240 rows): no seasonal, curtailment, or multi-turbine behaviour. Benchmark precision of 1.0 mostly reflects clean synthetic data and clear-cut injected faults, so don't expect that in the field.
- Subtle or slow drift (bearing wear over weeks) is the main miss; the detector is point-based and has no trend model.
- Wind baseline is a cubic fit over 3–9 m/s only; there is no cut-in/rated/cut-out curve, so high-wind behaviour is unmodelled.
- Curtailment, grid limits and maintenance outages look like underperformance; the playbooks tell the operator to check, but there's no curtailment input.
- Underperformance at zero-power hours is unobservable. The model is fitted on the same data it scores (transductive), with no time-based holdout.
- Likely causes are playbook-ranked hypotheses, not diagnoses. The 11 playbooks are written by me for this exercise, not vendor/OEM guidance.
- The optional LLM path and `/api/analyze_upload` are guarded but not exercised against a live LLM.

## Production-readiness gaps / next steps
Real labelled incidents and a time-based validation; per-asset-model baselines with cut-in/rated curves and curtailment flags; streaming ingestion and scheduled retraining; drift/trend detection (CUSUM, EWMA) for slow degradation; alert de-duplication and CMMS (work-order) integration; auth, rate limiting, logging/monitoring; OEM-vetted playbook corpus with embedding retrieval; operator feedback loop to tune thresholds.

## Secrets
None in the repo. `ANTHROPIC_API_KEY`, `LLM_MODEL`, `USE_LLM` are read from environment variables only.
