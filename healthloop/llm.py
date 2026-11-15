"""SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".

Language-model features. The LLM explains and summarises; it never decides alert levels.

Three jobs:
  ask()       Crew assistant. Answers an astronaut's question grounded in their live readings.
  extract()   Turns a free-text check-in into structured fields (a few bytes on the link).
  briefing()  Shift handover for the flight surgeon, from the data Earth actually received.

Where the model runs:
  Ship  -> a local model through Ollama (no internet in deep space). Used by ask() and extract().
  Earth -> Claude through the Anthropic API, if a key is set. Used by briefing().
  Either side falls back to the other kind of model, then to built-in rules, so the demo never breaks.

Configure with environment variables:
  HEALTHLOOP_SHIP_LLM / HEALTHLOOP_EARTH_LLM = auto | ollama | claude | none
  OLLAMA_HOST (default http://localhost:11434), HEALTHLOOP_OLLAMA_MODEL (default llama3.2:3b)
  ANTHROPIC_API_KEY, HEALTHLOOP_CLAUDE_MODEL (default claude-haiku-5-5)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

SYMPTOMS = ["Headache", "Nausea", "Congestion", "Back pain", "Vision change"]


# ------------------------------------------------------------------ providers
class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str: ...


class OllamaProvider:
    """A local open model, as it would run on the spacecraft's own computer."""

    def __init__(self, model: str | None = None, host: str | None = None, transport: httpx.BaseTransport | None = None):
        self.model = model or os.getenv("HEALTHLOOP_OLLAMA_MODEL", "llama3.2:3b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.http = httpx.Client(base_url=self.host, timeout=60, transport=transport)
        self.name = f"Local model ({self.model} via Ollama)"

    def available(self) -> bool:
        try:
            return self.http.get("/api/tags", timeout=1.5).status_code == 200
        except httpx.HTTPError:
            return False

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        r = self.http.post("/api/chat", json={
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.2, "num_predict": max_tokens}})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class ClaudeProvider:
    """Claude through the Anthropic API, as Mission Control on Earth could use."""

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("HEALTHLOOP_CLAUDE_MODEL", "claude-haiku-5-5")
        self._client = client
        self.name = f"Claude ({self.model})"

    def available(self) -> bool:
        return self._client is not None or bool(os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        msg = self._client.messages.create(model=self.model, max_tokens=max_tokens, system=system, temperature=0.2,
                                           messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def pick(preference: str, side: str) -> Provider | None:
    """Choose a provider. side='ship' prefers the local model, side='earth' prefers Claude."""
    pref = (preference or "auto").lower()
    if pref == "none":
        return None
    order = {"ollama": [OllamaProvider], "claude": [ClaudeProvider]}.get(
        pref, [OllamaProvider, ClaudeProvider] if side == "ship" else [ClaudeProvider, OllamaProvider])
    for cls in order:
        p = cls()
        if p.available():
            return p
    return None


# ------------------------------------------------------------------ prompts
ASK_SYSTEM = """You are HealthLoop, the onboard health assistant on a long-duration spacecraft.
Answer the astronaut's question using ONLY the data inside <health_data>.
Rules:
- You do not diagnose and you do not change alert levels. The detector sets levels; you explain them.
- The astronaut is not a doctor. Use everyday words, the way a friend who is a nurse would talk.
  Avoid medical terms; if you must use one, explain it in brackets the first time.
- Say "recovery score" for HRV and "beats a minute" for heart rate.
- Quote numbers next to the person's usual value, e.g. "78 beats a minute (usually about 64)".
- Only recommend steps listed under "actions", or simple self-care like rest, water and sleep.
- If anything sounds serious (chest pain, breathlessness, fainting, confusion), tell them to contact
  {contact} now and follow the onboard checklist.
- If the data does not answer the question, say so plainly.
- Second person, short sentences, under 120 words. No headings."""

EXTRACT_SYSTEM = """You convert an astronaut's free-text health check-in into JSON.
Return ONLY a JSON object with these keys:
  "fatigue": integer 1-5 (1 = fully rested, 5 = exhausted) or null if not mentioned
  "mood": integer 1-5 (1 = very low, 5 = very good) or null if not mentioned
  "symptoms": list containing only items from %s
  "summary": the check-in in at most 12 words
Do not guess symptoms that are not mentioned.""" % json.dumps(SYMPTOMS)

BRIEF_SYSTEM = """You write the shift handover for a flight surgeon in Mission Control.
Use ONLY the data inside <received_data>. It arrived over a delayed deep-space link, so every value is old by its data age.
Format:
- One line per crew member: name, status, anything abnormal with its value, and data age.
- Then "Open items:" listing unacknowledged alerts and anything to follow up, or "none".
- Say clearly if any crew member's data is more than 90 minutes old.
Under 150 words. Plain text, no markdown headings. Do not invent values or diagnoses."""


# ------------------------------------------------------------------ results
@dataclass
class Result:
    text: str
    source: str
    data: dict | None = None


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


def _clean_extraction(d: dict) -> dict:
    """Never trust model output blindly: clamp ranges and keep only known symptoms."""
    def level(v):
        try:
            return max(1, min(5, int(v))) if v is not None else None
        except (TypeError, ValueError):
            return None
    syms = [s for s in SYMPTOMS if any(str(x).strip().lower() == s.lower() for x in d.get("symptoms") or [])]
    return {"fatigue": level(d.get("fatigue")), "mood": level(d.get("mood")), "symptoms": syms,
            "summary": str(d.get("summary") or "")[:100]}


# ------------------------------------------------------------------ rule-based fallbacks
def _rules_ask(ctx: dict, question: str) -> str:
    q = question.lower()
    fs = ctx["findings"]
    if any(w in q for w in ("exercise", "workout", "train", "treadmill", "run")):
        if any(f["signal"] in ("hr", "hrv", "sleep", "spo2") for f in fs):
            return "Not hard exercise today. " + " ".join(f"{f['title']}." for f in fs[:2]) + \
                   " Do light activity only and re-check after resting."
        return "Your readings are within your normal range, so your planned exercise is fine."
    if fs:
        f = fs[0]
        return f"{f['title']}. {f['explanation']} Next step: {f['actions'][0]}"
    v = ctx["vitals"]
    return ("Everything is within your normal range right now: " +
            ", ".join(f"{x['label']} {x['value']} (usual {x['usual']})" for x in v.values()) + ".")


def _rules_extract(text: str) -> dict:
    t = text.lower()
    keys = {"Headache": ("headache", "head hurts", "migraine"), "Nausea": ("nause", "sick", "queasy", "vomit"),
            "Congestion": ("congest", "stuffy", "blocked nose", "sinus"), "Back pain": ("back pain", "back hurts", "back ache", "backache"),
            "Vision change": ("vision", "blurry", "blurred", "eyesight")}
    syms = [s for s, ws in keys.items() if any(w in t for w in ws)]
    fatigue = 5 if any(w in t for w in ("exhausted", "barely slept", "no sleep", "wiped")) else \
              4 if any(w in t for w in ("tired", "fatigue", "drained")) else \
              2 if any(w in t for w in ("rested", "energetic", "slept well")) else None
    mood = 2 if any(w in t for w in ("anxious", "down", "stressed", "sad", "lonely", "irritable")) else \
           4 if any(w in t for w in ("good", "happy", "great", "fine")) else None
    return {"fatigue": fatigue, "mood": mood, "symptoms": syms, "summary": text.strip()[:60]}


def _rules_brief(ctx: dict) -> str:
    lines = [f"{c['name']}: {c['status']}, HR {c['hr']}, SpO2 {c['spo2']}%, stress {c['stress']} (data age {c['data_age_min']} min)"
             + (" STALE" if c["data_age_min"] > 90 else "") for c in ctx["crew"]]
    open_items = [f"{a['crew']} {a['signal']} {a['level']} ({a['value']})" for a in ctx["alerts"] if not a["acknowledged"]]
    return "\n".join(lines) + "\nOpen items: " + ("; ".join(open_items) if open_items else "none")


# ------------------------------------------------------------------ the three jobs
class HealthAI:
    def __init__(self, ship: Provider | None = "auto", earth: Provider | None = "auto"):
        self.ship = pick(os.getenv("HEALTHLOOP_SHIP_LLM", "auto"), "ship") if ship == "auto" else ship
        self.earth = pick(os.getenv("HEALTHLOOP_EARTH_LLM", "auto"), "earth") if earth == "auto" else earth

    def status(self) -> dict:
        return {"ship": self.ship.name if self.ship else "Built-in rules (no model running)",
                "earth": self.earth.name if self.earth else "Built-in rules (no model running)"}

    def ask(self, ctx: dict, question: str) -> Result:
        question = question.strip()[:500]
        if self.ship:
            try:
                system = ASK_SYSTEM.format(contact="the commander" if ctx["is_medical_officer"] else "the crew medical officer")
                user = f"<health_data>\n{json.dumps(ctx, indent=1)}\n</health_data>\n\nQuestion from {ctx['name']}: {question}"
                return Result(self.ship.complete(system, user, 300), self.ship.name)
            except Exception as e:                       # model down mid-demo -> rules
                return Result(_rules_ask(ctx, question), f"Built-in rules (model error: {type(e).__name__})")
        return Result(_rules_ask(ctx, question), "Built-in rules (no model running)")

    def extract(self, text: str) -> Result:
        text = text.strip()[:600]
        if self.ship:
            try:
                d = _clean_extraction(_parse_json(self.ship.complete(EXTRACT_SYSTEM, text, 200)))
                return Result(d["summary"], self.ship.name, d)
            except Exception as e:
                d = _rules_extract(text)
                return Result(d["summary"], f"Built-in rules (model error: {type(e).__name__})", d)
        d = _rules_extract(text)
        return Result(d["summary"], "Built-in rules (no model running)", d)

    def briefing(self, ctx: dict) -> Result:
        if not ctx["crew"]:
            return Result("No data has reached Mission Control yet.", "none")
        if self.earth:
            try:
                return Result(self.earth.complete(BRIEF_SYSTEM, f"<received_data>\n{json.dumps(ctx, indent=1)}\n</received_data>", 400),
                              self.earth.name)
            except Exception as e:
                return Result(_rules_brief(ctx), f"Built-in rules (model error: {type(e).__name__})")
        return Result(_rules_brief(ctx), "Built-in rules (no model running)")
