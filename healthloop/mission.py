"""SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".

The whole mission in one object: crew simulator + detection + the deep-space link + Mission Control.

This is the Python twin of the web dashboard's built-in engine. `state()` returns a
snapshot in exactly the shape the dashboard renders, so the page can run on either engine.

Link rules:
  - Alerts (priority 0) jump the queue, then check-ins and all-clears (1), then telemetry (2).
  - Each step the link earns `bw` bytes/min x 10 min of budget (capped), and sends what fits.
  - If more than 12 telemetry packets are waiting, the oldest is dropped. Alerts are never dropped.
  - A sent packet arrives `delay` minutes later. During a blackout nothing is sent; packets wait onboard.
  - Acknowledgments from Earth travel back with the same delay.
"""
from __future__ import annotations

import struct
from collections import deque

from .detection import ALERT, LEVEL_NAMES
from .packets import Sealer, TelemetryRecord, decode, encode_alert, encode_checkin, encode_telemetry, json_size
from .simulate import MINUTES_PER_STEP, SCENARIOS, CrewSim, clock

# Each signal is tied to a NASA Human Research Program risk (nasa.gov/human-system-risks)
# and one of NASA's 5 Hazards of Human Spaceflight (nasa.gov/hrp/hazards).
NASA_RISK = {
    "hr": ("Risk of Cardiovascular Adaptations Contributing to Adverse Mission Performance and Health Outcomes", "Gravity fields"),
    "hrv": ("Risk of Cardiovascular Adaptations Contributing to Adverse Mission Performance and Health Outcomes", "Gravity fields"),
    "spo2": ("Risk of Reduced Crew Health and Performance Due to Hypoxia", "Hostile/closed environments"),
    "stress": ("Risk of Adverse Cognitive or Behavioral Changes and Psychiatric Disorders", "Isolation and confinement"),
    "sleep": ("Risk of Performance Decrements and Adverse Health Outcomes Resulting from Sleep Loss, "
              "Circadian Desynchronization, and Work Overload", "Isolation and confinement"),
}
SIGNAL_LABELS = {"hr": "Heart rate", "hrv": "Recovery (HRV)", "spo2": "Blood oxygen", "stress": "Stress score", "sleep": "Sleep"}
PLAIN_LABELS = {"hr": "Heart rate", "hrv": "Recovery", "spo2": "Blood oxygen", "stress": "Stress"}
SYMPTOMS = ["Headache", "Nausea", "Congestion", "Back pain", "Vision change"]
ACTIONS = {
    "hr": ["Stop what you're doing and sit down for 10 minutes, then check again.", "Drink a glass of water and check that the cabin isn't too warm.",
           "If you have chest pain, feel dizzy, or your heart feels like it's racing or skipping, open the heart emergency checklist and call the crew medical officer now."],
    "hrv": ["Your body needs more rest than usual. Do light exercise today instead of weights.", "Make sure you get your full sleep tonight."],
    "spo2": ["Check the oxygen and carbon dioxide readings on the life-support panel.", "Move to a module with good air flow, and push the finger sensor firmly back on.",
             "If it stays at 92% or lower, or you feel short of breath, put on the oxygen mask and tell the commander."],
    "stress": ["Take a 15-minute break from your task list.", "Book a private call with family or the crew psychologist.",
               "If this lasts more than a day, talk to the crew medical officer."],
    "sleep": ["Ask the commander to start your first task an hour later tomorrow.", "Turn down screens and cabin lights an hour before bed."],
}


def hm(t: int) -> str:
    m = (t % 144) * MINUTES_PER_STEP
    return f"{m // 60:02d}:{m % 60:02d}"


def describe(f, last_night: float | None = None) -> tuple[str, str]:
    """Plain-language title and explanation for the crew (no medical jargon)."""
    r = round
    if f.signal == "hr":
        return (f"Heart rate {r(f.value)} beats a minute",
                f"That's {r(f.value - f.baseline)} faster than usual for you at this time of day (usually about {r(f.baseline)}). "
                f"It has been this way for {f.minutes} minutes.")
    if f.signal == "hrv":
        return ("Your body isn't recovering as well as usual",
                f"Your recovery score is {r(f.value)}; it's usually about {r(f.baseline)}. "
                "A lower score means your body is under more strain than normal.")
    if f.signal == "spo2":
        return (f"Blood oxygen {f.value:.1f}%",
                f"It's usually {f.baseline:.1f}% for you. Below 94% needs attention. It could be the sensor, the cabin air, or your breathing.")
    if f.signal == "stress":
        return (f"Stress is high: {r(f.value)} out of 100",
                f"It's usually about {r(f.baseline)} for you, and it has stayed high for {f.minutes} minutes.")
    return (f"Short on sleep: {f.value:.1f} hours a night",
            f"That's your average over the last 3 nights. Last night was {last_night or 0:.1f} hours; you usually get about {f.baseline:.1f}.")


class Packet:
    __slots__ = ("kind", "prio", "t", "wire", "size", "json", "desc", "status", "sent", "sent_min", "arrive")

    def __init__(self, kind, prio, t, wire, json_len, desc):
        self.kind, self.prio, self.t, self.wire, self.size, self.json, self.desc = kind, prio, t, wire, len(wire), json_len, desc
        self.status, self.sent, self.sent_min, self.arrive = "Queued", None, None, None


class Mission:
    def __init__(self, seed: int = 20261105, delay: int = 12, bw: int = 6):
        self.sim = CrewSim(seed=seed)
        self.sealer = Sealer()
        self.delay, self.bw, self.blackout = delay, bw, False
        self.queue: list[Packet] = []
        self.down: list[Packet] = []
        self.up_queue: list[dict] = []
        self.up: list[dict] = []
        self.budget = 0.0
        self.sent = self.wire = self.json = self.dropped = 0
        self.pkts: deque[Packet] = deque(maxlen=14)
        self.mcc_crew: dict[int, dict] = {}
        self.mcc_alerts: list[dict] = []
        self.first_rx = None
        self.alert_seq = 1
        self.ship_log: list[dict] = []
        self.crew_alerts: dict[int, list[dict]] = {c.id: [] for c in self.sim.crew}
        self.acc = {c.id: {"n": 0, "hr": 0.0, "hrv": 0.0, "spo2": 0.0, "stress": 0.0} for c in self.sim.crew}

    @property
    def t(self) -> int:
        return self.sim.t

    # ---------------- onboard ----------------
    def log(self, who, text, lvl=0):
        self.ship_log.insert(0, dict(T=self.t, who=who.name if who else "", text=text, lvl=lvl))
        del self.ship_log[40:]

    def step(self, n: int = 1):
        for _ in range(n):
            for c, tr in self.sim.step():
                self._transition(c, tr)
            for c in self.sim.crew:
                a = self.acc[c.id]
                a["n"] += 1
                for k in ("hr", "hrv", "spo2", "stress"):
                    a[k] += c.last[k]
            if self.t % 3 == 0:
                self._routine()
            self._link_tick()

    def _transition(self, c, tr):
        f = tr.finding
        if tr.new > tr.old and f:
            title, _ = describe(f, c.nights[-1] if c.nights else None)
            aid = self.alert_seq
            self.alert_seq += 1
            self.crew_alerts[c.id].insert(0, dict(id=aid, sig=tr.signal, lvl=tr.new, T=self.t, title=title, ack=None))
            del self.crew_alerts[c.id][20:]
            self.log(c, ("ALERT" if tr.new == ALERT else "Watch") + ": " + title, tr.new)
            self._enqueue("Alert", 0, encode_alert(self.t, aid, c.id, tr.signal, tr.new, f.value),
                          f"{LEVEL_NAMES[tr.new]} · {c.name.split()[0]} · {SIGNAL_LABELS[tr.signal]}")
        elif tr.new == 0 and tr.old > 0:
            self.log(c, f"{SIGNAL_LABELS[tr.signal]} back within baseline", 0)
            self._enqueue("Clear", 1, encode_alert(self.t, 0, c.id, tr.signal, 0, 0),
                          f"Cleared · {c.name.split()[0]} · {SIGNAL_LABELS[tr.signal]}")

    def _routine(self):
        if not self.acc[0]["n"]:
            return
        recs = []
        for c in self.sim.crew:
            a = self.acc[c.id]
            n = a["n"]
            recs.append(TelemetryRecord(c.id, a["hr"] / n, a["hrv"] / n, a["spo2"] / n, a["stress"] / n,
                                        c.detector.level, c.nights[-1] if c.nights else 0))
            self.acc[c.id] = {"n": 0, "hr": 0.0, "hrv": 0.0, "spo2": 0.0, "stress": 0.0}
        self._enqueue("Telemetry", 2, encode_telemetry(self.t, recs), "4 crew · 30-min average")

    def checkin(self, crew_id: int, fatigue: int, mood: int, symptoms: int = 0):
        c = self.sim.crew[crew_id]
        c.checkin = (fatigue - 3) * 7 + (3 - mood) * 7
        self._enqueue("Check-in", 1, encode_checkin(self.t, c.id, fatigue, mood, symptoms), f"Check-in · {c.name.split()[0]}")
        self.log(c, f"Check-in submitted (fatigue {fatigue}, mood {mood})")

    def inject(self, crew_id: int, scenario: str) -> str:
        label = self.sim.inject(crew_id, scenario)
        self.log(None, f"Simulator: {label} injected for {self.sim.crew[crew_id].name}")
        return label

    # ---------------- link ----------------
    def set_link(self, delay=None, bw=None, blackout=None):
        if delay is not None:
            self.delay = int(delay)
        if bw is not None:
            self.bw = int(bw)
        if blackout is not None and bool(blackout) != self.blackout:
            self.blackout = bool(blackout)
            self.log(None, "Earth link lost. Storing packets onboard." if self.blackout else "Earth link restored. Forwarding stored packets.")

    def _enqueue(self, kind, prio, plain: bytes, desc: str):
        self.queue.append(Packet(kind, prio, self.t, self.sealer.seal(plain), json_size(decode(plain)), desc))
        routine = [p for p in self.queue if p.kind == "Telemetry"]
        if len(routine) > 12:
            old = routine[0]
            self.queue.remove(old)
            old.status = "Dropped (stale)"
            self.dropped += 1

    def _link_tick(self):
        now = self.t * MINUTES_PER_STEP
        for p in [p for p in self.down if p.arrive <= now]:
            self.down.remove(p)
            self._deliver(p)
        for u in [u for u in self.up if u["arrive"] <= now]:
            self.up.remove(u)
            self._ship_receive(u)
        if self.blackout:
            return
        cap = max(self.bw * MINUTES_PER_STEP * 3, 128)
        self.budget = min(self.budget + self.bw * MINUTES_PER_STEP, cap)
        self.queue.sort(key=lambda p: (p.prio, p.t))
        while self.queue and self.queue[0].size <= self.budget:
            p = self.queue.pop(0)
            self.budget -= p.size
            p.sent, p.sent_min, p.arrive, p.status = self.t, now, now + self.delay, "In flight"
            self.down.append(p)
            self.sent += 1
            self.wire += p.size
            self.json += p.json
            self.pkts.appendleft(p)
        while self.up_queue:
            u = self.up_queue.pop(0)
            u["sentMin"], u["arrive"] = now, now + self.delay
            self.up.append(u)

    def _deliver(self, p: Packet):
        d = decode(self.sealer.open(p.wire))         # raises if tampered
        p.status = "Delivered " + hm(self.t)
        if self.first_rx is None:
            self.first_rx = self.t
        if d["type"] == "telemetry":
            for r in d["records"]:
                self.mcc_crew[r["crew"]] = dict(hr=r["hr"], hrv=r["hrv"], spo2=r["spo2"], stress=r["stress"],
                                                lvl=r["level"], sleep=r["sleep"], asOf=d["t"])
        elif d["type"] == "alert":
            if d["level"] == 0:
                self.mcc_alerts.insert(0, dict(id=f"c{d['t']}{d['crew']}{d['signal']}", crew=d["crew"], sig=d["signal"],
                                               lvl=-1, created=d["t"], rx=self.t, clear=True))
            else:
                self.mcc_alerts.insert(0, dict(id=d["id"], crew=d["crew"], sig=d["signal"], lvl=d["level"], val=d["value"],
                                               created=d["t"], rx=self.t, ack=None))
                m = self.mcc_crew.get(d["crew"])
                if m and d["level"] > m["lvl"]:
                    m["lvl"] = d["level"]
        elif d["type"] == "checkin":
            self.mcc_alerts.insert(0, dict(id=f"r{d['t']}{d['crew']}", crew=d["crew"], report=True, fatigue=d["fatigue"],
                                           mood=d["mood"], sym=d["symptoms"], created=d["t"], rx=self.t, lvl=0))
        del self.mcc_alerts[30:]

    def ack(self, alert_id: int) -> bool:
        a = next((x for x in self.mcc_alerts if x.get("id") == alert_id and not x.get("clear") and not x.get("report")), None)
        if not a or a["ack"] is not None:
            return False
        a["ack"] = self.t
        self.up_queue.append(dict(id=alert_id, sentT=self.t, wire=self.sealer.seal(struct.pack(">BIH", 4, self.t, alert_id))))
        return True

    def _ship_receive(self, u):
        for c in self.sim.crew:
            for a in self.crew_alerts[c.id]:
                if a["id"] == u["id"]:
                    a["ack"] = dict(sent=u["sentT"], rx=self.t)
                    self.log(c, f"Flight surgeon acknowledged “{a['title']}” (sent {hm(u['sentT'])}, received {hm(self.t)})")

    # ---------------- snapshot for the dashboard ----------------
    def state(self) -> dict:
        crew = []
        for c in self.sim.crew:
            d = c.detector
            findings = []
            for f in c.findings:
                item = dict(sig=f.signal, lvl=f.level, x=f.value, m=f.baseline, sd=f.sd, mins=f.minutes)
                if f.signal == "sleep":
                    item["last"] = c.nights[-1]
                findings.append(item)
            sig_lvl = {k: s.level for k, s in d.state.items()}
            sig_lvl["sleep"] = d.sleep_level
            crew.append(dict(
                id=c.id, name=c.name, role=c.role, base=c.base, level=d.level, findings=findings, sigLvl=sig_lvl,
                series={k: [dict(v=v, m=m, s=s) for v, m, s in list(ser)[-48:]] for k, ser in c.series.items()},
                nights=[dict(T=t, h=h) for t, h in zip(c.night_t, c.nights)], alerts=self.crew_alerts[c.id][:6]))
        return dict(
            T=self.t, engine="python", crew=crew, shipLog=self.ship_log[:10],
            link=dict(delay=self.delay, bw=self.bw, blackout=self.blackout, queue=[{}] * len(self.queue),
                      down=[dict(prio=p.prio, sentMin=p.sent_min, arrive=p.arrive) for p in self.down],
                      up=[dict(sentMin=u["sentMin"], arrive=u["arrive"]) for u in self.up],
                      sent=self.sent, wire=self.wire, json=self.json, dropped=self.dropped,
                      pkts=[dict(sent=p.sent, kind=p.kind, desc=p.desc, json=p.json, size=p.size,
                                 hex=" ".join(f"{b:02x}" for b in p.wire[12:22]), status=p.status) for p in self.pkts]),
            mcc=dict(crew=self.mcc_crew, alerts=self.mcc_alerts[:10], firstRx=self.first_rx))

    def crew_detail(self, cid: int) -> dict:
        c = self.sim.crew[cid]
        out = []
        for f in c.findings:
            title, body = describe(f, c.nights[-1] if c.nights else None)
            out.append(dict(signal=f.signal, level=f.level_name, value=round(f.value, 2), baseline=round(f.baseline, 2),
                            z=round(f.z, 2), title=title, explanation=body, actions=ACTIONS[f.signal]))
        return dict(id=c.id, name=c.name, role=c.role, time=clock(self.t), level=LEVEL_NAMES[c.detector.level],
                    vitals={k: round(v, 1) for k, v in c.last.items()}, nights=[round(h, 1) for h in c.nights[-7:]], findings=out)

    # ---------------- what the language model is allowed to see ----------------
    def ai_context(self, cid: int) -> dict:
        """Onboard context for one astronaut: their live readings, baselines and findings."""
        from .simulate import is_asleep
        c = self.sim.crew[cid]
        ph = "asleep" if is_asleep(self.t) else "awake"
        units = {"hr": "beats a minute", "hrv": "(recovery score, higher = more rested)", "spo2": "%", "stress": "out of 100"}
        vitals = {}
        for k, v in c.last.items():
            b = c.detector.bl[ph][k]
            sd = max(b.sd, c.detector.signals[k].sd_floor)
            vitals[k] = dict(label=PLAIN_LABELS[k], value=f"{v:.1f} {units[k]}", usual=f"{b.m - 2 * sd:.1f} to {b.m + 2 * sd:.1f}",
                             level=LEVEL_NAMES[c.detector.state[k].level])
        findings = []
        for f in c.findings:
            title, body = describe(f, c.nights[-1] if c.nights else None)
            usual = f"{f.baseline:.1f} h" if f.signal == "sleep" else f"{f.baseline:.1f} {units[f.signal]}"
            findings.append(dict(signal=f.signal, level=f.level_name, title=title, usual=usual, explanation=body, actions=ACTIONS[f.signal],
                                 nasa_risk=NASA_RISK[f.signal][0]))
        return dict(name=c.name, role=c.role, is_medical_officer=c.role == "Crew Medical Officer", mission_time=clock(self.t),
                    phase=ph, overall_status=LEVEL_NAMES[c.detector.level], vitals=vitals, findings=findings,
                    sleep_last_nights_h=[round(h, 1) for h in c.nights[-3:]],
                    recent_alerts=[dict(title=a["title"], acknowledged_by_earth=bool(a["ack"])) for a in self.crew_alerts[cid][:3]],
                    earth_delay_min=self.delay, link="blackout" if self.blackout else "up")

    def earth_context(self) -> dict:
        """Mission Control context: only what has arrived over the link, with data ages."""
        names = {c.id: c.name for c in self.sim.crew}
        crew = [dict(name=names[i], status=LEVEL_NAMES[m["lvl"]], hr=m["hr"], hrv=m["hrv"], spo2=round(m["spo2"], 1),
                     stress=m["stress"], slept_h=m["sleep"], data_age_min=(self.t - m["asOf"]) * MINUTES_PER_STEP)
                for i, m in sorted(self.mcc_crew.items())]
        alerts = [dict(crew=names[a["crew"]], signal=SIGNAL_LABELS[a["sig"]], level=LEVEL_NAMES[a["lvl"]], value=a["val"],
                       nasa_risk=NASA_RISK[a["sig"]][0], nasa_hazard=NASA_RISK[a["sig"]][1],
                       happened=clock(a["created"]), acknowledged=a["ack"] is not None)
                  for a in self.mcc_alerts if not a.get("clear") and not a.get("report")][:8]
        checkins = [dict(crew=names[a["crew"]], fatigue=a["fatigue"], mood=a["mood"],
                         symptoms=[s for i, s in enumerate(SYMPTOMS) if a["sym"] & (1 << i)]) for a in self.mcc_alerts if a.get("report")][:4]
        return dict(earth_time=clock(self.t), one_way_delay_min=self.delay, link="blackout" if self.blackout else "up",
                    crew=crew, alerts=alerts, recent_checkins=checkins)
