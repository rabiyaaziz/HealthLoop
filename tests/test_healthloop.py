# SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
# directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".
import numpy as np
import pandas as pd
import pytest
from cryptography.exceptions import InvalidTag
from fastapi.testclient import TestClient

from healthloop.datasets import ECG_SIGNALS, ecg_rows, load_csv, load_fitbit
from healthloop.detection import ALERT, NOMINAL, WATCH, Detector
from healthloop.packets import Sealer, TelemetryRecord, decode, encode_alert, encode_telemetry, json_size
from healthloop.simulate import CrewSim


def hr_detector(**kw):
    return Detector.from_values({"awake": {"hr": (64, 2.6)}, "asleep": {"hr": (54, 2.5)}}, **kw)


# ---------- detection rules (the worked example from the docs) ----------
def test_single_spike_is_ignored():
    d = hr_detector()
    d.update({"hr": 80})
    f, _ = d.update({"hr": 64})
    assert not f


def test_three_high_readings_alert():
    d = hr_detector()
    levels = [d.update({"hr": x})[0] for x in (74, 74.6, 74.1)]
    assert levels[0] == [] and levels[1] == []
    assert levels[2][0].level == ALERT


def test_moderate_rise_is_watch():
    d = hr_detector()
    for _ in range(3):
        f, _ = d.update({"hr": 71})          # z ~ 2.7
    assert f[0].level == WATCH


def test_baseline_frozen_during_anomaly():
    d = hr_detector()
    for _ in range(20):
        d.update({"hr": 80})
    assert d.bl["awake"]["hr"].m == pytest.approx(64)


def test_hysteresis_needs_three_calm_readings():
    d = hr_detector()
    for _ in range(3):
        d.update({"hr": 76})
    assert d.update({"hr": 64})[0]          # still flagged
    assert d.update({"hr": 64})[0]
    assert not d.update({"hr": 64})[0]      # cleared on 3rd calm reading


def test_sleep_vs_wake_baselines():
    d = hr_detector()
    for _ in range(3):
        f, _ = d.update({"hr": 64}, asleep=True)   # normal awake, high for sleep (z=4)
    assert f and f[0].level == ALERT


def test_spo2_hard_limit_is_immediate():
    d = Detector.from_values({"awake": {"spo2": (98, .4)}})
    f, _ = d.update({"spo2": 91.5})
    assert f[0].level == ALERT


def test_sleep_rules():
    d = hr_detector()
    assert d.check_sleep([7, 7, 7, 7.2])[0] is None
    assert d.check_sleep([7, 7, 7, 4.8])[0].level == WATCH
    assert d.check_sleep([7, 7, 4.6, 4.5, 4.4])[0].level == ALERT


# ---------- simulation ----------
def test_no_false_alarms_over_a_week():
    sim = CrewSim(seed=7)
    events = [tr for _ in range(7 * 144) for _, tr in sim.step() if tr.new]
    assert events == []


@pytest.mark.parametrize("scenario,signal,within", [("cardiac", "hr", 18), ("hypoxia", "spo2", 6), ("isolation", "stress", 40)])
def test_scenarios_are_caught(scenario, signal, within):
    sim = CrewSim(seed=3)
    sim.inject(1, scenario)
    for step in range(within):
        if any(c.id == 1 and tr.signal == signal and tr.new for c, tr in sim.step()):
            return
    pytest.fail(f"{scenario} not detected within {within} steps")


# ---------- packets ----------
def test_packet_round_trip_and_size():
    recs = [TelemetryRecord(i, 64.4, 48.2, 97.8, 31, 0, 6.8) for i in range(4)]
    plain = encode_telemetry(12345, recs)
    assert len(plain) == 34
    s = Sealer()
    wire = s.seal(plain)
    assert len(wire) == 34 + Sealer.OVERHEAD
    back = decode(s.open(wire))
    assert back["records"][2]["hr"] == 64 and back["records"][2]["spo2"] == pytest.approx(97.8)
    assert json_size(back) > 4 * len(wire)


def test_tampering_is_detected():
    s = Sealer()
    wire = bytearray(s.seal(encode_alert(1, 7, 2, "spo2", 2, 91.5)))
    wire[15] ^= 1
    with pytest.raises(InvalidTag):
        s.open(bytes(wire))


# ---------- real & external data ----------
def test_real_ecg_burst_flagged():
    rows = ecg_rows()
    d = Detector.fit(rows[:6], signals=ECG_SIGNALS, minutes_per_sample=0, persist=2)
    flagged = [r["time"] for r in rows if any(f.signal == "ectopy" for f in d.update(r)[0])]
    assert flagged and all(130 <= t <= 220 for t in flagged)


def test_fitbit_adapter(tmp_path):
    t = pd.date_range("2016-04-12", periods=3 * 24 * 60, freq="1min")
    hr = 60 + 3 * np.random.default_rng(0).standard_normal(len(t))
    hr[-60:] += 30                               # last hour: big rise
    pd.DataFrame({"Id": 1503960366, "Time": t.strftime("%-m/%-d/%Y %-I:%M:%S %p"), "Value": hr.round()}) \
        .to_csv(tmp_path / "heartrate_seconds_merged.csv", index=False)
    pd.DataFrame({"Id": [1503960366] * 3, "SleepDay": ["4/12/2016", "4/13/2016", "4/14/2016"],
                  "TotalSleepRecords": 1, "TotalMinutesAsleep": [420, 300, 270], "TotalTimeInBed": 480}) \
        .to_csv(tmp_path / "sleepDay_merged.csv", index=False)
    uid, rows, nights = load_fitbit(tmp_path)
    assert uid == 1503960366 and len(rows) == 3 * 144 and nights == [7.0, 5.0, 4.5]
    from healthloop.detection import SIGNALS
    d = Detector.fit(rows[:144], signals={"hr": SIGNALS["hr"]})
    assert any(d.update(r, r["asleep"])[0] for r in rows[-12:])


def test_generic_csv(tmp_path):
    t = pd.date_range("2026-01-01", periods=200, freq="10min")
    pd.DataFrame({"timestamp": t, "hr": 60.0, "spo2": 98.0}).to_csv(tmp_path / "x.csv", index=False)
    rows = load_csv(tmp_path / "x.csv")
    assert len(rows) == 200 and rows[0]["hr"] == 60.0


# ---------- API + mission ----------
def test_api_flow():
    from healthloop import api
    c = TestClient(api.app)
    c.post("/api/reset")
    assert "HealthLoop" in c.get("/").text
    s = c.get("/api/state").json()
    assert s["engine"] == "python" and len(s["crew"]) == 4
    assert c.post("/api/inject", json={"crew": 1, "scenario": "hypoxia"}).status_code == 200
    s = c.post("/api/step", params={"n": 12}).json()
    mateo = s["crew"][1]
    assert mateo["sigLvl"]["spo2"] == 2 and mateo["alerts"][0]["sig"] in ("spo2", "hr")
    assert len(mateo["series"]["spo2"]) == 48
    # alert reaches Earth after the light-time delay, is acknowledged, ack returns to the ship
    alert = next(a for a in s["mcc"]["alerts"] if a.get("lvl") == 2)
    assert c.post(f"/api/ack/{alert['id']}").status_code == 200
    s = c.post("/api/step", params={"n": 3}).json()
    assert any(a["ack"] for a in s["crew"][1]["alerts"])
    assert s["link"]["wire"] < s["link"]["json"]
    assert c.get("/api/crew/1").json()["findings"][0]["actions"]
    assert c.get("/api/downlink").json()["packets"]
    assert c.post("/api/inject", json={"crew": 9, "scenario": "hypoxia"}).status_code == 404
    assert c.post("/api/checkin", json={"crew": 0, "fatigue": 9, "mood": 3}).status_code == 422


def test_blackout_stores_and_forwards():
    from healthloop.mission import Mission
    m = Mission(seed=5)
    m.set_link(blackout=True)
    m.inject(2, "hypoxia")
    m.step(12)
    assert m.sent == 0 and any(p.kind == "Alert" for p in m.queue)
    m.set_link(blackout=False)
    m.step(4)
    assert any(a.get("sig") == "spo2" for a in m.mcc_alerts)


def test_low_bandwidth_drops_only_telemetry():
    from healthloop.mission import Mission
    m = Mission(seed=5, bw=1)
    m.inject(0, "cardiac")
    m.step(144)
    assert m.dropped > 0
    assert all(p.kind == "Telemetry" for p in m.queue if p.status.startswith("Dropped"))
    assert any(a.get("sig") == "hr" for a in m.mcc_alerts)
