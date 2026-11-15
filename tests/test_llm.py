# SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
# directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from healthloop.llm import ClaudeProvider, HealthAI, OllamaProvider, _clean_extraction
from healthloop.mission import Mission


class Fake:
    """Stands in for a model: records the prompt, returns a canned reply."""
    name = "Fake model"

    def __init__(self, reply="ok", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def complete(self, system, user, max_tokens=400):
        self.calls.append((system, user))
        if self.fail:
            raise TimeoutError("model down")
        return self.reply


@pytest.fixture
def flagged():
    m = Mission(seed=4)
    m.inject(1, "cardiac")
    m.step(30)
    return m


def test_ask_is_grounded_in_live_data(flagged):
    fake = Fake("Your heart rate is 85 bpm (usual 64). Rest for 10 minutes.")
    r = HealthAI(ship=fake, earth=None).ask(flagged.ai_context(1), "Why am I flagged?")
    system, user = fake.calls[0]
    assert r.source == "Fake model" and "85 bpm" in r.text
    assert "do not diagnose" in system and "the commander" in system      # Mateo is the medical officer
    data = json.loads(user.split("<health_data>")[1].split("</health_data>")[0])
    assert data["findings"][0]["signal"] in ("hr", "hrv") and data["findings"][0]["actions"]


def test_model_failure_falls_back_to_rules(flagged):
    r = HealthAI(ship=Fake(fail=True), earth=None).ask(flagged.ai_context(1), "Can I exercise today?")
    assert r.text.startswith("Not hard exercise") and "rules" in r.source


def test_extraction_is_validated():
    fake = Fake('Sure! {"fatigue": 9, "mood": "2", "symptoms": ["headache", "Fever", "Back pain"], "summary": "tired, sore"}')
    r = HealthAI(ship=fake, earth=None).extract("so tired, headache and back pain")
    assert r.data == {"fatigue": 5, "mood": 2, "symptoms": ["Headache", "Back pain"], "summary": "tired, sore"}


def test_bad_json_falls_back_to_rules():
    r = HealthAI(ship=Fake("I think you have a headache."), earth=None).extract("barely slept, stuffy nose and headache")
    assert r.data["symptoms"] == ["Headache", "Congestion"] and r.data["fatigue"] == 5 and "rules" in r.source


def test_briefing_uses_only_received_data():
    m = Mission(seed=4)
    fake = Fake("Mateo: Alert.")
    ai = HealthAI(ship=None, earth=fake)
    assert ai.briefing(m.earth_context()).text.startswith("No data")    # nothing has arrived yet
    m.inject(1, "hypoxia")
    m.step(12)
    ctx = m.earth_context()
    assert ai.briefing(ctx).source == "Fake model"
    sent = fake.calls[0][1]
    assert "data_age_min" in sent and "Blood oxygen" in sent


def test_ollama_provider_request_shape():
    seen = {}

    def handler(req):
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " hello "}})
    p = OllamaProvider(model="llama3.2:3b", transport=httpx.MockTransport(handler))
    assert p.available() and p.complete("sys", "hi") == "hello"
    assert seen["model"] == "llama3.2:3b" and seen["stream"] is False and seen["messages"][0]["role"] == "system"


def test_claude_provider_request_shape():
    class Block:
        type, text = "text", "brief"

    class Msgs:
        def create(self, **kw):
            self.kw = kw
            return type("M", (), {"content": [Block()]})()
    client = type("C", (), {"messages": Msgs()})()
    p = ClaudeProvider(model="claude-haiku-5-5", client=client)
    assert p.complete("sys", "data") == "brief"
    kw = client.messages.kw
    assert kw["model"] == "claude-haiku-5-5" and kw["system"] == "sys" and kw["messages"][0]["content"] == "data"


def test_api_ai_endpoints():
    from healthloop import api
    api.ai = HealthAI(ship=None, earth=None)
    c = TestClient(api.app)
    c.post("/api/reset")
    c.post("/api/inject", json={"crew": 2, "scenario": "hypoxia"})
    c.post("/api/step", params={"n": 12})
    a = c.post("/api/ask", json={"crew": 2, "question": "Why am I flagged?"}).json()
    assert "Blood oxygen" in a["answer"] or "Heart rate" in a["answer"]
    r = c.post("/api/checkin", json={"crew": 2, "fatigue": 3, "mood": 3, "text": "exhausted and a headache"}).json()
    assert r["fatigue"] == 5 and r["symptoms"] == ["Headache"]
    assert "Open items" in c.get("/api/briefing").json()["briefing"]
    assert c.post("/api/ask", json={"crew": 2, "question": ""}).status_code == 422
