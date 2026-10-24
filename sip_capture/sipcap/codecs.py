"""음성 코덱 디코더.

- G.711 mu-law (PCMU, PT 0) / A-law (PCMA, PT 8): 순수 Python (ITU-T G.711 표준 알고리즘)
- G.729 / G.729A / G.729B (PT 18): 외부 디코더 사용
    1) bcg729 라이브러리 (bcg729.dll) — SID(comfort noise) 프레임까지 처리
    2) ffmpeg (ffmpeg.exe) — bcg729 가 없을 때 사용, SID 프레임은 무음 처리

모든 디코더는 8000 Hz, 16-bit signed little-endian mono PCM(bytes)을 돌려준다.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
from array import array
from pathlib import Path

SAMPLE_RATE = 8000

# RFC 3551 정적 payload type (동적 PT 는 SDP a=rtpmap 으로 알아냄)
STATIC_PAYLOAD_TYPES = {
    0: "PCMU",
    3: "GSM",
    4: "G723",
    8: "PCMA",
    9: "G722",
    13: "CN",
    18: "G729",
}

# 디코딩 가능한 코덱 이름 (rtpmap 이름은 대소문자/변형이 섞여 들어옴)
_CODEC_ALIASES = {
    "PCMU": "PCMU",
    "G711U": "PCMU",
    "PCMA": "PCMA",
    "G711A": "PCMA",
    "G729": "G729",
    "G729A": "G729",
    "G729B": "G729",
    "G729AB": "G729",
}

# 음성이 아닌 payload (오디오 복원 시 건너뜀)
NON_AUDIO = {"CN", "TELEPHONE-EVENT", "RED", "ULPFEC"}

G729_FRAME_BYTES = 10      # 10 ms 음성 프레임
G729_SID_BYTES = 2         # Annex B SID(무음 구간 comfort noise) 프레임
G729_FRAME_SAMPLES = 80


def normalize_codec(name: str | None) -> str | None:
    """'g729a', 'PCMU' 같은 rtpmap 이름 → 'G729' / 'PCMU' / 'PCMA'. 지원 안 하면 대문자 원래 이름."""
    if not name:
        return None
    key = name.strip().upper().replace(".", "").replace("-", "")
    return _CODEC_ALIASES.get(key, name.strip().upper())


def is_supported(codec: str | None) -> bool:
    return codec in ("PCMU", "PCMA", "G729")


# ---------------------------------------------------------------------------
# G.711
# ---------------------------------------------------------------------------

def _ulaw_to_linear(u: int) -> int:
    u = ~u & 0xFF
    t = ((u & 0x0F) << 3) + 0x84
    t <<= (u & 0x70) >> 4
    return (0x84 - t) if (u & 0x80) else (t - 0x84)


def _alaw_to_linear(a: int) -> int:
    a ^= 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    return t if (a & 0x80) else -t


_ULAW_TABLE = [_ulaw_to_linear(i) for i in range(256)]
_ALAW_TABLE = [_alaw_to_linear(i) for i in range(256)]


def _table_decode(payload: bytes, table: list[int]) -> bytes:
    out = array("h", [table[b] for b in payload])
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def decode_pcmu(payload: bytes) -> bytes:
    return _table_decode(payload, _ULAW_TABLE)


def decode_pcma(payload: bytes) -> bytes:
    return _table_decode(payload, _ALAW_TABLE)


# ---------------------------------------------------------------------------
# G.729
# ---------------------------------------------------------------------------

def split_g729_payload(payload: bytes) -> list[bytes]:
    """RTP payload 하나 → 10바이트 음성 프레임들 (+ 끝에 2바이트 SID 가 붙을 수 있음, RFC 3551 4.5.6)."""
    n_full = len(payload) // G729_FRAME_BYTES
    frames = [payload[i * 10:(i + 1) * 10] for i in range(n_full)]
    rest = payload[n_full * 10:]
    if len(rest) == G729_SID_BYTES:
        frames.append(rest)
    return frames


def _app_dirs() -> list[Path]:
    """외부 도구(dll/exe)를 찾을 폴더: 실행 파일 옆, 프로젝트 폴더, bin/."""
    dirs = []
    if getattr(sys, "frozen", False):          # PyInstaller exe
        dirs.append(Path(sys.executable).parent)
        if hasattr(sys, "_MEIPASS"):
            dirs.append(Path(sys._MEIPASS))
    here = Path(__file__).resolve().parent.parent
    dirs += [here, here / "bin"]
    return dirs


class Bcg729Decoder:
    """bcg729 (Belledonne Communications) 공유 라이브러리를 ctypes 로 호출. bcg729 1.x API."""

    name = "bcg729"

    def __init__(self, lib_path: str):
        self._lib = ctypes.CDLL(lib_path)
        self._lib.initBcg729DecoderChannel.restype = ctypes.c_void_p
        self._lib.initBcg729DecoderChannel.argtypes = []
        self._lib.closeBcg729DecoderChannel.argtypes = [ctypes.c_void_p]
        self._lib.bcg729Decoder.restype = None
        self._lib.bcg729Decoder.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.POINTER(ctypes.c_int16),
        ]
        self.path = lib_path

    @classmethod
    def find(cls) -> "Bcg729Decoder | None":
        candidates = []
        env = os.environ.get("SIPCAP_BCG729")
        if env:
            candidates.append(env)
        for d in _app_dirs():
            for fname in ("bcg729.dll", "libbcg729.dll", "libbcg729.so", "libbcg729.so.0", "libbcg729.dylib"):
                p = d / fname
                if p.exists():
                    candidates.append(str(p))
        found = ctypes.util.find_library("bcg729")
        if found:
            candidates.append(found)
        for c in candidates:
            try:
                return cls(c)
            except (OSError, AttributeError):
                continue
        return None

    def decode(self, frames: list[bytes]) -> list[bytes]:
        """프레임 리스트 → 프레임마다 80샘플(160바이트) PCM. 스트림 하나당 한 번 호출 (디코더 상태 유지)."""
        ctx = self._lib.initBcg729DecoderChannel()
        out_buf = (ctypes.c_int16 * G729_FRAME_SAMPLES)()
        result = []
        try:
            for f in frames:
                sid = 1 if len(f) == G729_SID_BYTES else 0
                self._lib.bcg729Decoder(ctx, f, len(f), 0, sid, 0, out_buf)
                pcm = array("h", out_buf)
                if sys.byteorder != "little":
                    pcm.byteswap()
                result.append(pcm.tobytes())
        finally:
            self._lib.closeBcg729DecoderChannel(ctx)
        return result


class FfmpegG729Decoder:
    """ffmpeg 내장 G.729 디코더 사용. 10바이트 프레임만 이어붙여 한 번에 디코딩."""

    name = "ffmpeg"

    def __init__(self, exe: str):
        self.path = exe

    @classmethod
    def find(cls) -> "FfmpegG729Decoder | None":
        candidates = []
        env = os.environ.get("SIPCAP_FFMPEG")
        if env:
            candidates.append(env)
        for d in _app_dirs():
            for fname in ("ffmpeg.exe", "ffmpeg"):
                p = d / fname
                if p.exists():
                    candidates.append(str(p))
        w = shutil.which("ffmpeg")
        if w:
            candidates.append(w)
        for c in candidates:
            if os.path.isfile(c):
                return cls(c)
        return None

    def decode(self, frames: list[bytes]) -> list[bytes]:
        voice = [f for f in frames if len(f) == G729_FRAME_BYTES]
        if not voice:
            return [bytes(G729_FRAME_SAMPLES * 2) for _ in frames]
        flags = 0x08000000 if os.name == "nt" else 0     # CREATE_NO_WINDOW: GUI 에서 콘솔 창 안 뜨게
        proc = subprocess.run(
            [self.path, "-hide_banner", "-loglevel", "error", "-f", "g729", "-i", "pipe:0",
             "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
            input=b"".join(voice), capture_output=True, creationflags=flags, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg G.729 디코딩 실패: " + proc.stderr.decode(errors="replace").strip())
        pcm = proc.stdout
        step = G729_FRAME_SAMPLES * 2
        silence = bytes(step)
        result, k = [], 0
        for f in frames:
            if len(f) == G729_FRAME_BYTES:
                chunk = pcm[k * step:(k + 1) * step]
                result.append(chunk + bytes(step - len(chunk)))
                k += 1
            else:
                result.append(silence)              # SID 프레임 → 무음
        return result


_g729_cache: list = []


def get_g729_decoder(refresh: bool = False):
    """사용 가능한 G.729 디코더 (bcg729 우선, 없으면 ffmpeg). 둘 다 없으면 None."""
    if refresh or not _g729_cache:
        _g729_cache[:] = [Bcg729Decoder.find() or FfmpegG729Decoder.find()]
    return _g729_cache[0]


def g729_decoder_status() -> str:
    d = get_g729_decoder()
    if d is None:
        return "G.729 디코더 없음 (bcg729.dll 또는 ffmpeg.exe 필요)"
    return f"G.729 디코더: {d.name} ({d.path})"
