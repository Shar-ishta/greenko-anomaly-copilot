import pandas as pd
from backend.pipeline import load_data, run_pipeline, extract_events, retrieve_playbooks


def test_bundled_events():
    f = run_pipeline(); ev = extract_events(f)
    got = {(e["asset_id"], e["start_id"], e["end_id"]) for e in ev}
    assert got == {("SOLAR_INV_01", 67, 69), ("WTG_07", 173, 174)}
    assert all(e["severity"] == "high" for e in ev)


def test_no_false_alarms_on_clean_rows():
    f = run_pipeline()
    assert int(f.is_anomaly.sum()) == 5


def test_explanation_has_actions_and_safety():
    for e in extract_events(run_pipeline()):
        x = e["explanation"]
        assert x["next_checks"] and x["likely_causes"] and x["safety_note"]


def test_bad_data_flagged():
    df = load_data(); i = df.index[(df.asset_id == "WTG_07")][50]
    df.loc[i, "power_mw"] = -3.0
    assert run_pipeline(df).loc[i, "is_anomaly"]


def test_missing_columns_rejected():
    import pytest
    with pytest.raises(ValueError):
        load_data(pd.DataFrame({"a": [1]}))
