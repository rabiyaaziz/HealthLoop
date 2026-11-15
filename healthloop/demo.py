"""SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".

Command-line demos that print a timeline and save a chart.

  python -m healthloop.demo simulate --scenario cardiac --crew 1
  python -m healthloop.demo ecg
  python -m healthloop.demo fitbit path/to/fitbit_folder
  python -m healthloop.demo csv my_data.csv
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .datasets import ECG_SIGNALS, ecg_beats, ecg_rows, load_csv, load_fitbit
from .detection import LEVEL_NAMES, SIGNALS, Detector
from .simulate import SCENARIOS, CrewSim, clock, is_asleep

COL = {0: "#1C8456", 1: "#A86F12", 2: "#C1352C"}


def _shade(ax, xs, levels):
    for i, lv in enumerate(levels):
        if lv and i + 1 < len(xs):
            ax.axvspan(xs[i], xs[i + 1], color=COL[lv], alpha=.15, lw=0)


def run_rows(det: Detector, rows: list[dict], label=str):
    out = []
    for r in rows:
        findings, trans = det.update(r, asleep=r.get("asleep", False))
        out.append(findings)
        for tr in trans:
            what = LEVEL_NAMES[tr.new].upper() if tr.new else "clear"
            extra = f" value {tr.finding.value:.2f} vs usual {tr.finding.baseline:.2f} (z={tr.finding.z:.1f})" if tr.finding else ""
            print(f"  {label(r['time'])}  {tr.signal:<7} -> {what}{extra}")
    return out


def cmd_simulate(a):
    sim = CrewSim()
    c = sim.crew[a.crew]
    print(f"Injecting '{SCENARIOS[a.scenario][0]}' for {c.name} at {clock(sim.t)}")
    ts, hr, m, sd, lv = [], [], [], [], []
    for i in range(a.hours * 6):
        for who, tr in sim.step():
            if who is c:
                f = tr.finding
                print(f"  {clock(sim.t)}  {tr.signal:<7} -> {LEVEL_NAMES[tr.new].upper() if tr.new else 'clear'}"
                      + (f"  {f.value:.1f} vs usual {f.baseline:.1f}" if f else ""))
        if i == 6:
            sim.inject(a.crew, a.scenario)
        b = c.detector.bl["asleep" if is_asleep(sim.t) else "awake"][a.signal]
        ts.append(i * 10 / 60); hr.append(c.last[a.signal]); m.append(b.m); sd.append(max(b.sd, 1e-6))
        lv.append(c.detector.state[a.signal].level)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(ts, [x - 2 * s for x, s in zip(m, sd)], [x + 2 * s for x, s in zip(m, sd)], color="#2556C9", alpha=.12, label="personal normal range (±2 sd)")
    ax.plot(ts, hr, color="#14202B", lw=1.4, label=a.signal)
    _shade(ax, ts, lv)
    ax.set(xlabel="hours", ylabel=a.signal, title=f"{c.name}: {SCENARIOS[a.scenario][0]} (shaded = Watch / Alert)")
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout()
    fig.savefig(a.out); print("saved", a.out)


def cmd_ecg(a):
    rows = ecg_rows()
    base = rows[:6]                                   # first 60 s = this patient's baseline
    det = Detector.fit(base, signals=ECG_SIGNALS, minutes_per_sample=0, persist=2)  # 2 windows = 20 s of beats
    print(f"Real ECG, MIT-BIH record 208: {len(rows)} windows of 10 s. Baseline from first 60 s.")
    results = run_rows(det, rows, label=lambda t: f"{int(t)//60}:{int(t)%60:02d}")
    lv = [max([f.level for f in fs if f.signal == 'ectopy'], default=0) for fs in results]
    ecg, _, peaks = ecg_beats()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    t = [i / 360 for i in range(len(ecg))]
    a1.plot(t, ecg, lw=.4, color="#14202B"); a1.plot(peaks / 360, ecg[peaks], "o", ms=2, color="#2556C9")
    a1.set(ylabel="mV", title="Real ECG with detected heartbeats (MIT-BIH record 208)")
    xs = [r["time"] for r in rows] + [rows[-1]["time"] + 10]
    a2.step(xs[:-1], [r["ectopy"] for r in rows], where="post", color="#14202B")
    _shade(a2, xs, lv); _shade(a1, xs, lv)
    a2.set(xlabel="seconds", ylabel="irregular beat share", title="HealthLoop flags (shaded = Watch / Alert)")
    fig.tight_layout(); fig.savefig(a.out); print("saved", a.out)


def cmd_fitbit(a):
    uid, rows, nights = load_fitbit(a.path, a.user)
    n = len(rows) // 3
    det = Detector.fit(rows[:n], signals={"hr": SIGNALS["hr"]})
    print(f"Fitbit user {uid}: {len(rows)} ten-minute samples. Baseline from first third.")
    run_rows(det, rows[n:], label=lambda t: str(t)[:16])
    if nights:
        f, _ = det.check_sleep(nights)
        print("  sleep:", f"{f.level_name} ({f.value:.1f} h avg)" if f else "Nominal")


def cmd_csv(a):
    rows = load_csv(a.path, a.time_col)
    n = max(len(rows) // 3, 10)
    det = Detector.fit(rows[:n])
    print(f"{a.path}: {len(rows)} samples. Baseline from first {n}.")
    run_rows(det, rows[n:], label=lambda t: str(t)[:16])


def main():
    p = argparse.ArgumentParser(prog="healthloop.demo")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("simulate"); s.add_argument("--scenario", choices=SCENARIOS, default="cardiac")
    s.add_argument("--crew", type=int, default=1); s.add_argument("--signal", default="hr")
    s.add_argument("--hours", type=int, default=16); s.add_argument("--out", default="simulate.png"); s.set_defaults(f=cmd_simulate)
    e = sp.add_parser("ecg"); e.add_argument("--out", default="ecg.png"); e.set_defaults(f=cmd_ecg)
    f = sp.add_parser("fitbit"); f.add_argument("path"); f.add_argument("--user", type=int); f.set_defaults(f=cmd_fitbit)
    c = sp.add_parser("csv"); c.add_argument("path"); c.add_argument("--time-col", default="timestamp"); c.set_defaults(f=cmd_csv)
    a = p.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
