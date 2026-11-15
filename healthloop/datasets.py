"""SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".

Load real data into HealthLoop's format: a list of rows
{"time": Timestamp, "asleep": bool, "hr": .., "hrv": .., ...}.

Three sources:
  ecg_rows()          real ECG bundled with SciPy (MIT-BIH Arrhythmia DB, record 208)
  load_fitbit(folder) the "Fitbit Fitness Tracker Data" export from Kaggle
  load_csv(path)      any CSV with a timestamp column plus hr / hrv / spo2 / stress
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .detection import Signal

# ---------------------------------------------------------------- ECG
ECTOPY = Signal("ectopy", "Irregular beats", "fraction", +1, 0.03)
ECG_SIGNALS = {
    "hr": Signal("hr", "Heart rate", "bpm", +1, 2.0),
    "ectopy": ECTOPY,
}


def ecg_beats(fs: int = 360):
    """Return (ecg, filtered, R-peak sample indices) from SciPy's real ECG recording."""
    from scipy import datasets, signal

    ecg = datasets.electrocardiogram()                       # 5 min, 360 Hz, millivolts
    b, a = signal.butter(2, [5 / (fs / 2), 15 / (fs / 2)], "band")   # keep QRS energy
    filt = signal.filtfilt(b, a, ecg)
    energy = np.abs(filt)
    peaks, _ = signal.find_peaks(energy, distance=int(0.25 * fs), height=np.percentile(energy, 98) * 0.4)
    return ecg, filt, peaks


def ecg_rows(window_s: int = 10, fs: int = 360) -> list[dict]:
    """Split the recording into windows; per window compute heart rate and the share of
    beats whose RR interval is more than 20 % off the window median (irregular beats)."""
    _, _, peaks = ecg_beats(fs)
    rr = np.diff(peaks) / fs
    t = peaks[1:] / fs
    rows = []
    for start in range(0, int(t[-1]), window_s):
        r = rr[(t >= start) & (t < start + window_s)]
        if len(r) < 3:
            continue
        med = np.median(r)
        rows.append({
            "time": start, "asleep": False,
            "hr": float(60 / r.mean()),
            "ectopy": float(np.mean(np.abs(r - med) / med > 0.2)),
            "beats": int(len(r)),
        })
    return rows


# ---------------------------------------------------------------- Fitbit (Kaggle)
def load_fitbit(folder: str | Path, user_id: int | None = None, freq: str = "10min"):
    """Kaggle 'Fitbit Fitness Tracker Data' (Mobius). Uses
       heartrate_seconds_merged.csv  (Id, Time, Value)
       minuteSleep_merged.csv        (Id, date, value, logId)  optional, marks sleep
       sleepDay_merged.csv           (Id, SleepDay, TotalMinutesAsleep, ...) optional, nightly totals
    Returns (user_id, rows, nights_hours)."""
    folder = Path(folder)
    hr = pd.read_csv(_find(folder, "heartrate_seconds_merged.csv"))
    hr["Time"] = pd.to_datetime(hr["Time"], format="mixed")
    if user_id is None:
        user_id = int(hr["Id"].value_counts().idxmax())     # person with the most data
    hr = hr[hr["Id"] == user_id].set_index("Time")["Value"].resample(freq).mean().dropna()

    asleep = pd.Series(False, index=hr.index)
    p = _find(folder, "minuteSleep_merged.csv", required=False)
    if p:
        ms = pd.read_csv(p)
        ms = ms[ms["Id"] == user_id]
        ms["date"] = pd.to_datetime(ms["date"], format="mixed")
        slept = ms.set_index("date")["value"].resample(freq).count() > 0
        asleep = slept.reindex(hr.index, fill_value=False)
    else:
        asleep = pd.Series([h >= 23 or h < 7 for h in hr.index.hour], index=hr.index)

    nights: list[float] = []
    p = _find(folder, "sleepDay_merged.csv", required=False)
    if p:
        sd = pd.read_csv(p)
        sd = sd[sd["Id"] == user_id].sort_values("SleepDay")
        nights = (sd["TotalMinutesAsleep"] / 60).round(2).tolist()

    rows = [{"time": ts, "asleep": bool(a), "hr": float(v)} for ts, v, a in zip(hr.index, hr.values, asleep.values)]
    return user_id, rows, nights


def load_csv(path: str | Path, time_col: str = "timestamp", freq: str = "10min",
             sleep_hours: tuple[int, int] = (23, 7)) -> list[dict]:
    """Generic: any CSV with a timestamp column plus some of hr, hrv, spo2, stress."""
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], format="mixed")
    cols = [c for c in ("hr", "hrv", "spo2", "stress") if c in df.columns]
    df = df.set_index(time_col)[cols].resample(freq).mean().dropna(how="all")
    lo, hi = sleep_hours
    return [{"time": ts, "asleep": bool(ts.hour >= lo or ts.hour < hi),
             **{c: (None if pd.isna(r[c]) else float(r[c])) for c in cols}} for ts, r in df.iterrows()]


def _find(folder: Path, name: str, required: bool = True) -> Path | None:
    hits = sorted(folder.rglob(name))
    if not hits and required:
        raise FileNotFoundError(f"{name} not found under {folder}")
    return hits[-1] if hits else None
