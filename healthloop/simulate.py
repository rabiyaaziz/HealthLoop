"""SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".

Simulated crew for testing and demos.

One step = 10 minutes. Crew sleep 23:00-07:00. Scenarios inject a fault
that ramps in, holds, then ramps out, exactly like the web simulator.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .detection import Detector, Finding, Transition

STEPS_PER_DAY = 144
MINUTES_PER_STEP = 10

CREW = [
    dict(id=0, name="Ines Okafor", role="Commander", hr=58, hrv=62, spo2=98.2, stress=28, sleep=7.1),
    dict(id=1, name="Mateo Varga", role="Crew Medical Officer", hr=64, hrv=48, spo2=97.8, stress=32, sleep=6.8),
    dict(id=2, name="Lin Haruki", role="Mission Specialist", hr=61, hrv=55, spo2=98.0, stress=25, sleep=7.4),
    dict(id=3, name="Priya Nandakumar", role="Flight Engineer", hr=67, hrv=44, spo2=97.6, stress=35, sleep=6.9),
]

# each effect: signal, magnitude, duration (steps), ramp (steps)
SCENARIOS = {
    "cardiac": ("Elevated resting heart rate", [("hr", 24, 72, 6), ("hrv", 0.3, 72, 6)]),
    "hypoxia": ("Low blood oxygen", [("spo2", 6.5, 30, 3), ("hr", 12, 30, 3)]),
    "isolation": ("Isolation stress", [("stress", 32, 144, 12), ("hrv", 0.2, 144, 12), ("hr", 7, 144, 12)]),
    "sleep": ("Sleep loss (3 nights)", [("sleep", 2.9, 432, 1), ("stress", 20, 432, 60), ("hrv", 0.25, 432, 60)]),
}


def is_asleep(t: int) -> bool:
    h = (t % STEPS_PER_DAY) / 6
    return h >= 23 or h < 7


def clock(t: int) -> str:
    m = (t % STEPS_PER_DAY) * MINUTES_PER_STEP
    return f"Day {t // STEPS_PER_DAY} {m // 60:02d}:{m % 60:02d}"


@dataclass
class Effect:
    signal: str
    mag: float
    dur: int
    ramp: int
    start: int

    def at(self, t: int) -> float:
        dt = t - self.start
        if dt < 0 or dt > self.dur:
            return 0.0
        r = min(1.0, dt / self.ramp)
        tail = self.dur - dt
        if tail < self.ramp:
            r = min(r, tail / self.ramp)
        return self.mag * r


@dataclass
class CrewMember:
    id: int
    name: str
    role: str
    base: dict
    detector: Detector
    effects: list[Effect] = field(default_factory=list)
    nights: list[float] = field(default_factory=list)
    night_t: list[int] = field(default_factory=list)
    series: dict = field(default_factory=lambda: {k: deque(maxlen=288) for k in ('hr', 'hrv', 'spo2', 'stress')})
    last: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    checkin: float = 0.0

    def eff(self, k: str, t: int) -> float:
        return sum(e.at(t) for e in self.effects if e.signal == k)


def _initial_detector(b: dict) -> Detector:
    return Detector.from_values({
        "awake": {"hr": (b["hr"], 3), "hrv": (b["hrv"], 5), "spo2": (b["spo2"], .4), "stress": (b["stress"], 5.5)},
        "asleep": {"hr": (b["hr"] * .84, 2.5), "hrv": (b["hrv"] * 1.3, 5.5), "spo2": (b["spo2"] - .3, .4), "stress": (b["stress"] - 12, 4.5)},
    })


class CrewSim:
    def __init__(self, seed: int = 20261105, warmup_days: int = 14, start_day: int = 87):
        self.rng = np.random.default_rng(seed)
        self.crew = [CrewMember(c["id"], c["name"], c["role"], c, _initial_detector(c)) for c in CREW]
        self.t = start_day * STEPS_PER_DAY + 48 - warmup_days * STEPS_PER_DAY
        for _ in range(warmup_days * STEPS_PER_DAY):   # learn baselines, discard events
            self.step()

    def inject(self, crew_id: int, scenario: str) -> str:
        label, fx = SCENARIOS[scenario]
        c = self.crew[crew_id]
        c.effects += [Effect(k, mag, dur, ramp, self.t) for k, mag, dur, ramp in fx]
        return label

    def _sample(self, c: CrewMember) -> dict:
        t, b, g = self.t, c.base, self.rng.standard_normal
        a = is_asleep(t)
        circ = math.sin(((t % STEPS_PER_DAY) / 6 - 15) / 24 * 2 * math.pi)
        return {
            "hr": float(np.clip((b["hr"] * .84 if a else b["hr"] + 2 * circ) + g() * 2.2 + c.eff("hr", t), 35, 200)),
            "hrv": float(np.clip((b["hrv"] * 1.3 if a else b["hrv"]) * (1 - c.eff("hrv", t)) + g() * 4, 5, 150)),
            "spo2": float(np.clip(b["spo2"] + (-.3 if a else 0) + g() * .35 - c.eff("spo2", t), 80, 100)),
            "stress": float(np.clip((b["stress"] - 12 if a else b["stress"] + 3 * circ) + g() * 4.5
                                    + c.eff("stress", t) + (0 if a else c.checkin), 0, 100)),
        }

    def step(self) -> list[tuple[CrewMember, Transition]]:
        """Advance 10 minutes. Returns (crew member, transition) for anything that changed level."""
        self.t += 1
        events = []
        for c in self.crew:
            if self.t % STEPS_PER_DAY == 42:   # 07:00 wake-up: record last night's sleep
                h = float(np.clip(c.base["sleep"] + self.rng.standard_normal() * .45 - c.eff("sleep", self.t), 2.5, 9.5))
                c.nights = (c.nights + [h])[-14:]
                c.night_t = (c.night_t + [self.t])[-14:]
            x = self._sample(c)
            c.last = x
            c.checkin *= .985
            ph = "asleep" if is_asleep(self.t) else "awake"
            for k, ser in c.series.items():        # value + the baseline it was judged against
                b = c.detector.bl[ph][k]
                ser.append((x[k], b.m, max(b.sd, c.detector.signals[k].sd_floor)))
            findings, trans = c.detector.update(x, asleep=is_asleep(self.t))
            sf, st = c.detector.check_sleep(c.nights)
            if sf:
                findings.append(sf)
            if st:
                trans.append(st)
            c.findings = sorted(findings, key=lambda f: -f.level)
            c.effects = [e for e in c.effects if self.t - e.start <= e.dur]
            events += [(c, tr) for tr in trans]
        return events
