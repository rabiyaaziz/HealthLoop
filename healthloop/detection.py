"""SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".

Personal-baseline anomaly detection.

Same rules as the web dashboard, so both give identical results:

1. Each astronaut has their OWN baseline per signal, kept separately for
   sleep and waking hours (a heart rate of 52 is normal asleep, odd awake).
2. Baseline = exponentially weighted mean (m) and variance (v).
3. Each reading gets a z-score: z = (x - m) / sd, signed so that a
   positive value always means "in the worrying direction".
4. Persistence: z > 2.5 for 3 readings in a row -> WATCH,
   and if z > 3.5 at that point -> ALERT. One odd reading is often a sensor glitch.
5. Hard limits fire immediately (SpO2 <= 92 %, HR >= 120 bpm).
6. Hysteresis: a level only drops after 3 calm readings (z < 1.5).
7. Abnormal readings are NOT learned, so the baseline can't drift toward the problem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

NOMINAL, WATCH, ALERT = 0, 1, 2
LEVEL_NAMES = ["Nominal", "Watch", "Alert"]


@dataclass(frozen=True)
class Signal:
    key: str
    label: str
    unit: str
    direction: int          # +1: high is bad, -1: low is bad
    sd_floor: float         # smallest sd we trust (avoids huge z on a very steady signal)
    hard_limit: Optional[Callable[[float], int]] = None  # returns a level from the raw value


def _spo2_limit(x: float) -> int:
    return ALERT if x <= 92 else WATCH if x <= 94 else NOMINAL


def _hr_limit(x: float) -> int:
    return ALERT if x >= 120 else NOMINAL


SIGNALS: dict[str, Signal] = {
    "hr": Signal("hr", "Heart rate", "bpm", +1, 2.0, _hr_limit),
    "hrv": Signal("hrv", "Recovery (HRV)", "ms", -1, 3.0),
    "spo2": Signal("spo2", "Blood oxygen", "%", -1, 0.35, _spo2_limit),
    "stress": Signal("stress", "Stress score", "/100", +1, 4.0),
}


@dataclass
class Baseline:
    m: float
    v: float

    @property
    def sd(self) -> float:
        return math.sqrt(self.v)

    def learn(self, x: float, alpha: float) -> None:
        d = x - self.m
        self.m += alpha * d
        self.v = (1 - alpha) * (self.v + alpha * d * d)


@dataclass
class Finding:
    signal: str
    level: int
    value: float
    baseline: float
    sd: float
    z: float
    minutes: int

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES[self.level]


@dataclass
class Transition:
    """A level change worth telling someone about (new alert, escalation, or all-clear)."""
    signal: str
    old: int
    new: int
    finding: Optional[Finding]


@dataclass
class _SigState:
    streak: int = 0
    calm: int = 0
    level: int = NOMINAL


class Detector:
    """Anomaly detector for ONE person.

    Typical use:
        det = Detector.from_values({"awake": {"hr": (64, 3)}, "asleep": {"hr": (54, 2.5)}})
        findings, transitions = det.update({"hr": 74}, asleep=False)
    """

    def __init__(
        self,
        baselines: dict[str, dict[str, Baseline]],
        signals: dict[str, Signal] | None = None,
        *,
        minutes_per_sample: int = 10,
        alpha: float = 0.02,
        watch_z: float = 2.5,
        alert_z: float = 3.5,
        persist: int = 3,
        clear_z: float = 1.5,
        learn_below_z: float = 2.0,
    ):
        self.bl = baselines
        self.signals = signals or SIGNALS
        self.minutes = minutes_per_sample
        self.alpha, self.watch_z, self.alert_z = alpha, watch_z, alert_z
        self.persist, self.clear_z, self.learn_below_z = persist, clear_z, learn_below_z
        self.state = {k: _SigState() for k in self.signals}
        self.sleep_level = NOMINAL

    # ---------- construction helpers ----------
    @classmethod
    def from_values(cls, init: dict[str, dict[str, tuple[float, float]]], **kw) -> "Detector":
        """init[phase][signal] = (mean, sd)"""
        bl = {ph: {k: Baseline(m, sd * sd) for k, (m, sd) in sigs.items()} for ph, sigs in init.items()}
        return cls(bl, **kw)

    @classmethod
    def fit(cls, rows: list[dict], signals: dict[str, Signal] | None = None, **kw) -> "Detector":
        """Learn a starting baseline from history rows: {"asleep": bool, "hr": .., ...}."""
        signals = signals or SIGNALS
        init: dict[str, dict[str, tuple[float, float]]] = {}
        for ph, flag in (("awake", False), ("asleep", True)):
            init[ph] = {}
            for k, s in signals.items():
                xs = [r[k] for r in rows if r.get("asleep", False) == flag and r.get(k) is not None]
                if len(xs) < 3:
                    continue
                m = sum(xs) / len(xs)
                sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
                init[ph][k] = (m, max(sd, s.sd_floor))
        if not init["asleep"]:
            init["asleep"] = dict(init["awake"])
        return cls.from_values(init, signals=signals, **kw)

    # ---------- main step ----------
    def update(self, reading: dict[str, float], asleep: bool = False) -> tuple[list[Finding], list[Transition]]:
        phase = "asleep" if asleep else "awake"
        findings: list[Finding] = []
        transitions: list[Transition] = []
        for k, sig in self.signals.items():
            x = reading.get(k)
            if x is None or k not in self.bl[phase]:
                continue
            b = self.bl[phase][k]
            sd = max(b.sd, sig.sd_floor)
            z = sig.direction * (x - b.m) / sd          # positive = worrying direction
            st = self.state[k]

            st.streak = st.streak + 1 if z > self.watch_z else 0
            level = NOMINAL
            if st.streak >= self.persist:
                level = ALERT if z > self.alert_z else WATCH
            if sig.hard_limit:
                level = max(level, sig.hard_limit(x))

            # hysteresis: only step down after `persist` calm readings
            if level < st.level:
                st.calm = st.calm + 1 if z < self.clear_z else 0
                if st.calm < self.persist:
                    level = st.level
            else:
                st.calm = 0

            f = None
            if level:
                f = Finding(k, level, x, b.m, sd, z, max(st.streak, 1) * self.minutes)
                findings.append(f)
            if level != st.level and (level > st.level or level == NOMINAL):
                transitions.append(Transition(k, st.level, level, f))
            st.level = level

            if z < self.learn_below_z and level == NOMINAL:
                b.learn(x, self.alpha)                   # only learn from normal data

        findings.sort(key=lambda f: -f.level)
        return findings, transitions

    def check_sleep(self, nights_hours: list[float]) -> tuple[Optional[Finding], Optional[Transition]]:
        """Sleep is judged once a day from the list of nightly totals (newest last)."""
        if not nights_hours:
            return None, None
        last3 = nights_hours[-3:]
        avg = sum(last3) / len(last3)
        prior = nights_hours[:-3]
        usual = sum(prior) / len(prior) if prior else avg
        level = NOMINAL
        if len(last3) >= 3 and avg < 5:
            level = ALERT
        elif (len(last3) >= 3 and avg < 6) or nights_hours[-1] < 5:
            level = WATCH
        f = Finding("sleep", level, avg, usual, 0.0, 0.0, 0) if level else None
        t = None
        if level != self.sleep_level and (level > self.sleep_level or level == NOMINAL):
            t = Transition("sleep", self.sleep_level, level, f)
        self.sleep_level = level
        return f, t

    @property
    def level(self) -> int:
        return max([s.level for s in self.state.values()] + [self.sleep_level])
