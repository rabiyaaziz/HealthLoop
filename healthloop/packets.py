"""SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".

Compact binary packets + AES-GCM sealing for the deep-space downlink.

Byte layouts are identical to the web dashboard, so either side can decode the other.

Telemetry (type 1):  [1][t:uint32][n:uint8] then n records of 7 bytes:
                     [crew][hr][hrv][(spo2-80)*10][stress][level][sleep*10]
Alert (type 2):      [2][t:uint32][alert_id:uint16][crew][signal_code][level][value*10:uint16]
Check-in (type 3):   [3][t:uint32][crew][fatigue][mood][symptom_bits]

Sealed packet on the wire = 12-byte nonce + ciphertext + 16-byte auth tag
(so +28 bytes). AES-GCM both hides the data and detects tampering.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SIGNAL_CODES = ["hr", "hrv", "spo2", "stress", "sleep"]
_clamp = lambda x, lo, hi: max(lo, min(hi, int(round(x))))


@dataclass
class TelemetryRecord:
    crew: int
    hr: float
    hrv: float
    spo2: float
    stress: float
    level: int
    sleep: float


def encode_telemetry(t: int, records: list[TelemetryRecord]) -> bytes:
    out = bytearray(struct.pack(">BIB", 1, t, len(records)))
    for r in records:
        out += bytes([r.crew, _clamp(r.hr, 0, 255), _clamp(r.hrv, 0, 255), _clamp((r.spo2 - 80) * 10, 0, 255),
                      _clamp(r.stress, 0, 100), r.level, _clamp(r.sleep * 10, 0, 255)])
    return bytes(out)


def encode_alert(t: int, alert_id: int, crew: int, signal: str, level: int, value: float) -> bytes:
    return struct.pack(">BIHBBBH", 2, t, alert_id, crew, SIGNAL_CODES.index(signal), level, _clamp(value * 10, 0, 65535))


def encode_checkin(t: int, crew: int, fatigue: int, mood: int, symptoms: int) -> bytes:
    return struct.pack(">BIBBBB", 3, t, crew, fatigue, mood, symptoms)


def decode(b: bytes) -> dict:
    kind = b[0]
    if kind == 1:
        _, t, n = struct.unpack(">BIB", b[:6])
        recs = []
        for i in range(n):
            c, hr, hrv, sp, st, lv, sl = b[6 + i * 7: 13 + i * 7]
            recs.append(dict(crew=c, hr=hr, hrv=hrv, spo2=sp / 10 + 80, stress=st, level=lv, sleep=sl / 10))
        return dict(type="telemetry", t=t, records=recs)
    if kind == 2:
        _, t, aid, c, code, lv, val = struct.unpack(">BIHBBBH", b)
        return dict(type="alert", t=t, id=aid, crew=c, signal=SIGNAL_CODES[code], level=lv, value=val / 10)
    if kind == 3:
        _, t, c, fa, mo, sy = struct.unpack(">BIBBBB", b)
        return dict(type="checkin", t=t, crew=c, fatigue=fa, mood=mo, symptoms=sy)
    raise ValueError(f"unknown packet type {kind}")


def json_size(decoded: dict) -> int:
    """What the same packet would cost as JSON, for the bandwidth comparison."""
    return len(json.dumps(decoded).encode())


class Sealer:
    OVERHEAD = 12 + 16

    def __init__(self, key: bytes | None = None):
        self.key = key or AESGCM.generate_key(bit_length=256)
        self._aes = AESGCM(self.key)

    def seal(self, plain: bytes, associated: bytes = b"healthloop/v1") -> bytes:
        nonce = os.urandom(12)                       # never reuse a nonce with the same key
        return nonce + self._aes.encrypt(nonce, plain, associated)

    def open(self, wire: bytes, associated: bytes = b"healthloop/v1") -> bytes:
        """Raises cryptography.exceptions.InvalidTag if anything was altered."""
        return self._aes.decrypt(wire[:12], wire[12:], associated)
