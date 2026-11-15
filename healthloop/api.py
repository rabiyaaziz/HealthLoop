"""SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".

HealthLoop server: the Python engine behind the dashboard.

Run:        uvicorn healthloop.api:app --reload
Dashboard:  http://127.0.0.1:8000/        (runs on this Python engine)
API docs:   http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import base64
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .llm import SYMPTOMS, HealthAI
from .mission import Mission
from .packets import decode
from .simulate import MINUTES_PER_STEP, SCENARIOS, clock

app = FastAPI(title="HealthLoop", version="0.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DASHBOARD = Path(__file__).parent / "static" / "index.html"

mission = Mission()
lock = threading.Lock()          # one mission, many requests
ai = HealthAI()                  # picks local model / Claude / rules at startup


class Inject(BaseModel):
    crew: int
    scenario: str


class Checkin(BaseModel):
    crew: int
    fatigue: int = Field(ge=1, le=5)
    mood: int = Field(ge=1, le=5)
    symptoms: int = Field(0, ge=0, le=31)
    text: str = Field("", max_length=600)


class Ask(BaseModel):
    crew: int
    question: str = Field(min_length=1, max_length=500)


class LinkSettings(BaseModel):
    delay: int | None = Field(None, ge=3, le=22)
    bw: int | None = Field(None, ge=1, le=40)
    blackout: bool | None = None


def _crew(cid: int):
    if not 0 <= cid < len(mission.sim.crew):
        raise HTTPException(404, "No crew member with that id")


@app.get("/", include_in_schema=False)
def dashboard():
    # no-store: browsers always fetch the current dashboard instead of a saved copy
    return FileResponse(DASHBOARD, headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def state():
    """Full snapshot in the shape the dashboard renders."""
    with lock:
        return mission.state()


@app.post("/api/step")
def step(n: int = 1):
    """Advance the mission by n steps of 10 minutes, then return the new state."""
    if not 1 <= n <= 1440:
        raise HTTPException(400, "n must be between 1 and 1440 (10 days)")
    with lock:
        mission.step(n)
        return mission.state()


@app.post("/api/inject")
def inject(body: Inject):
    if body.scenario not in SCENARIOS:
        raise HTTPException(400, f"scenario must be one of {list(SCENARIOS)}")
    _crew(body.crew)
    with lock:
        return dict(injected=mission.inject(body.crew, body.scenario), crew=mission.sim.crew[body.crew].name)


@app.post("/api/checkin")
def checkin(body: Checkin):
    """Sliders plus optional free text. The text is turned into structured fields by the onboard model;
    only those few bytes go to Earth, never the raw text."""
    _crew(body.crew)
    fatigue, mood, sym, understood = body.fatigue, body.mood, body.symptoms, None
    if body.text.strip():
        r = ai.extract(body.text)          # outside the lock: model calls can be slow
        d = r.data
        sym |= sum(1 << SYMPTOMS.index(s) for s in d["symptoms"])
        if d["fatigue"] and fatigue == 3:
            fatigue = d["fatigue"]
        if d["mood"] and mood == 3:
            mood = d["mood"]
        understood = dict(fields=d, source=r.source)
    with lock:
        mission.checkin(body.crew, fatigue, mood, sym)
    return dict(ok=True, fatigue=fatigue, mood=mood, symptoms=[s for i, s in enumerate(SYMPTOMS) if sym & (1 << i)],
                understood=understood)


@app.post("/api/ask")
def ask(body: Ask):
    """Crew assistant: answers grounded in the astronaut's live readings."""
    _crew(body.crew)
    with lock:
        ctx = mission.ai_context(body.crew)
    r = ai.ask(ctx, body.question)
    return dict(answer=r.text, source=r.source, time=ctx["mission_time"])


@app.get("/api/briefing")
def briefing():
    """Shift handover for the flight surgeon, from received data only."""
    with lock:
        ctx = mission.earth_context()
    r = ai.briefing(ctx)
    return dict(briefing=r.text, source=r.source, time=ctx["earth_time"])


@app.get("/api/ai")
def ai_status():
    return ai.status()


@app.post("/api/ack/{alert_id}")
def ack(alert_id: int):
    with lock:
        if not mission.ack(alert_id):
            raise HTTPException(404, "No unacknowledged alert with that id has reached Mission Control")
    return dict(ok=True)


@app.post("/api/link")
def link(body: LinkSettings):
    with lock:
        mission.set_link(body.delay, body.bw, body.blackout)
        return dict(delay=mission.delay, bw=mission.bw, blackout=mission.blackout)


@app.get("/api/crew/{cid}")
def crew_detail(cid: int):
    """One astronaut, with plain-language explanations and next steps for each finding."""
    _crew(cid)
    with lock:
        return mission.crew_detail(cid)


@app.get("/api/downlink")
def downlink():
    """Packets that reached Earth, with ciphertext and the decrypted contents."""
    with lock:
        delivered = [p for p in mission.pkts if p.status.startswith("Delivered")]
        return dict(delay_min=mission.delay, in_flight=len(mission.down), packets=[
            dict(sent=clock(p.sent), kind=p.kind, bytes_on_wire=p.size, as_json=p.json,
                 ciphertext_b64=base64.b64encode(p.wire).decode(), decoded=decode(mission.sealer.open(p.wire)))
            for p in delivered])


@app.post("/api/reset")
def reset():
    global mission
    with lock:
        mission = Mission()
    return dict(ok=True)
