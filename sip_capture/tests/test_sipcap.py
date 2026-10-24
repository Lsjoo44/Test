import shutil
import subprocess
import sys
import wave
from array import array
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from sipcap.codecs import (FfmpegG729Decoder, decode_pcma, decode_pcmu, normalize_codec,  # noqa: E402
                           split_g729_payload)
from sipcap.rtp import parse_rtp  # noqa: E402
from sipcap.sip import display_party, parse_sdp, parse_sip, parse_sip_stream  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def _ffmpeg_decode(fmt: str, data: bytes) -> bytes:
    return subprocess.run([FFMPEG, "-loglevel", "error", "-f", fmt, "-ar", "8000", "-ac", "1", "-i", "pipe:0",
                           "-f", "s16le", "pipe:1"], input=data, capture_output=True, check=True).stdout


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg 없음")
@pytest.mark.parametrize("fmt,fn", [("mulaw", decode_pcmu), ("alaw", decode_pcma)])
def test_g711_matches_ffmpeg(fmt, fn):
    data = bytes(range(256))
    assert fn(data) == _ffmpeg_decode(fmt, data)


def test_g711_encoders_roundtrip():
    from make_sample_pcap import lin2alaw, lin2ulaw
    for enc, dec in ((lin2ulaw, decode_pcmu), (lin2alaw, decode_pcma)):
        for x in range(-32000, 32000, 997):
            y = array("h", dec(bytes([enc(x)])))[0]
            assert abs(x - y) <= max(16, abs(x) // 16), (enc.__name__, x, y)


def test_sip_parse_compact_and_folding():
    raw = (b"INVITE sip:bob@b.com SIP/2.0\r\n"
           b"v: SIP/2.0/UDP a.com\r\n"
           b"f: \"Alice\" <sip:alice@a.com>;tag=123\r\n"
           b"t: <sip:bob@b.com>\r\n"
           b"i: abc@a.com\r\n"
           b"CSeq: 7 INVITE\r\n"
           b"Subject: long\r\n  folded\r\n"
           b"c: application/sdp\r\n"
           b"l: 4\r\n\r\nv=0\r\n")
    m = parse_sip(raw)
    assert m.method == "INVITE" and m.call_id == "abc@a.com" and m.cseq == (7, "INVITE")
    assert m.from_tag == "123" and m.to_tag is None
    assert m.header("subject") == "long folded"
    assert m.body == b"v=0\r"
    assert display_party(m.header("from")) == "Alice <alice>"


def test_sip_stream_partial_and_two_messages():
    one = b"SIP/2.0 180 Ringing\r\nCall-ID: x\r\nCSeq: 1 INVITE\r\nContent-Length: 0\r\n\r\n"
    two = b"SIP/2.0 200 OK\r\nCall-ID: x\r\nCSeq: 1 INVITE\r\nContent-Length: 3\r\n\r\nabc"
    assert parse_sip_stream(two[:-1]) == (None, 0)
    m, used = parse_sip_stream(one + two)
    assert m.status_code == 180 and used == len(one)
    m, used = parse_sip_stream(two)
    assert m.status_code == 200 and m.body == b"abc"


def test_sdp_media_level_connection():
    body = (b"v=0\r\no=- 1 1 IN IP4 1.1.1.1\r\ns=-\r\nc=IN IP4 1.1.1.1\r\nt=0 0\r\n"
            b"m=audio 4000 RTP/AVP 18 96 101\r\nc=IN IP4 2.2.2.2\r\na=rtpmap:96 G729a/8000\r\n"
            b"a=fmtp:18 annexb=no\r\nm=video 5000 RTP/AVP 99\r\n")
    a, v = parse_sdp(body)
    assert (a.media, a.ip, a.port, a.payload_types) == ("audio", "2.2.2.2", 4000, [18, 96, 101])
    assert normalize_codec(a.rtpmap[96]) == "G729" and a.fmtp[18] == "annexb=no"
    assert v.ip == "1.1.1.1"


def test_rtp_parse_csrc_ext_padding():
    import struct
    hdr = struct.pack("!BBHII", 0x80 | 0x20 | 0x10 | 1, 18, 5, 160, 99) + b"\0\0\0\1"  # 1 CSRC
    ext = b"\xbe\xde\x00\x01" + b"\1\2\3\4"
    pkt = parse_rtp(hdr + ext + b"P" * 10 + b"\0\0\3")
    assert pkt.payload_type == 18 and pkt.payload == b"P" * 10
    assert parse_rtp(bytes([0x80, 200]) + bytes(10)) is None      # RTCP SR


def test_split_g729_payload():
    assert [len(f) for f in split_g729_payload(bytes(22))] == [10, 10, 2]
    assert [len(f) for f in split_g729_payload(bytes(2))] == [2]


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg 없음")
def test_ffmpeg_g729_sample_count():
    dec = FfmpegG729Decoder(FFMPEG)
    frames = [bytes([i * 7 % 256] * 10) for i in range(50)] + [b"\x12\x34"] + [bytes(10)] * 5
    out = dec.decode(frames)
    assert len(out) == len(frames) and all(len(o) == 160 for o in out)
    assert out[50] == bytes(160)                                   # SID → 무음
    assert any(o != bytes(160) for o in out[:50])


# ---------------------------------------------------------------------------
# end-to-end: 가짜 pcap → 통화 4건 / WAV 확인
# ---------------------------------------------------------------------------

scapy = pytest.importorskip("scapy")


@pytest.fixture(scope="module")
def tracker(tmp_path_factory):
    from make_sample_pcap import make
    from sipcap.calls import CallTracker
    from sipcap.capture import read_pcap
    p = tmp_path_factory.mktemp("pcap") / "sample.pcap"
    make(str(p))
    t = CallTracker()
    read_pcap(str(p), t)
    return t


def _dominant_freq(samples) -> float:
    crossings = sum(1 for a, b in zip(samples, samples[1:]) if a < 0 <= b)
    return crossings * 8000 / len(samples)


def test_calls_detected(tracker):
    calls = tracker.call_list()
    assert [c.state for c in calls] == ["종료", "종료", "Busy(486)", "종료"]
    assert [c.codec for c in calls] == ["PCMU", "G729", "", "PCMA"]
    assert calls[3].from_ == "홍길동" and calls[0].to == "2002"
    assert not tracker.orphan_streams                              # RTCP 는 스트림이 아님


def test_pcmu_call_stats(tracker):
    c = tracker.call_list()[0]
    by_dir = {s.direction: s for s in c.streams}
    a, b = by_dir["caller->callee"], by_dir["callee->caller"]
    assert (len(a.packets), a.lost) == (149, 1)                    # 1개 손실, 순서 뒤바뀜은 손실 아님
    assert (len(b.packets), b.lost) == (150, 0)                    # seq/timestamp wrap 처리
    assert a.jitter_ms < 1.0


def test_pcmu_stereo_wav(tracker, tmp_path):
    from sipcap.export import call_to_stereo, save_call_wav
    c = tracker.call_list()[0]
    left, right = call_to_stereo(c)
    assert abs(len(left) - 150 * 160) <= 40
    assert abs(_dominant_freq(left) - 440) < 10
    assert abs(_dominant_freq(right) - 880) < 10
    assert all(v == 0 for v in left[50 * 160:51 * 160])            # 손실 구간 = 무음
    p = save_call_wav(c, tmp_path / "c.wav")
    with wave.open(str(p)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (2, 8000, 2)


def test_pcma_tcp_call(tracker):
    from sipcap.export import call_to_stereo
    c = tracker.call_list()[3]
    assert [e.start_line.split()[1] for e in c.events][:3] == ["sip:4004@192.168.0.20;transport=tcp", "180", "200"]
    left, right = call_to_stereo(c)
    assert abs(_dominant_freq(left) - 300) < 10 and abs(_dominant_freq(right) - 600) < 10


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg 없음")
def test_g729_call_fragmented_invite(tracker):
    from sipcap.export import stream_to_pcm
    c = tracker.call_list()[1]
    assert c.events[0].start_line.startswith("INVITE")               # 조각난 INVITE 재조립
    by_dir = {s.direction: s for s in c.streams}
    a = by_dir["caller->callee"]
    assert a.lost == 0 and len(a.packets) == 81                     # DTX 는 손실 아님
    pcm = stream_to_pcm(a, FfmpegG729Decoder(FFMPEG))
    assert len(pcm) == 100 * 160                                    # timestamp 기준 2초
    assert all(v == 0 for v in pcm[41 * 160:60 * 160])              # SID 이후 DTX 구간 무음
    assert any(v != 0 for v in pcm[:40 * 160])


def test_cli_read(tmp_path):
    from make_sample_pcap import make
    p = tmp_path / "s.pcap"
    make(str(p))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "sipcap_cli.py"), "read", str(p), "-o", str(out)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    wavs = sorted(x.name for x in out.glob("*.wav"))
    assert len(wavs) == (3 if FFMPEG else 2)
    assert (out / "calls.csv").read_text(encoding="utf-8-sig").count("\n") == 5
