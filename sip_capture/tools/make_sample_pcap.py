"""테스트용 가짜 SIP 통화 pcap 생성 (실제 전화기 없이 프로그램 동작 확인용).

  python tools/make_sample_pcap.py sample.pcap

포함된 통화
  1) G.711 mu-law (PCMU) / UDP  — 발신 440 Hz, 수신 880 Hz 사인파, 패킷 1개 손실 + 순서 뒤바뀜 1회
  2) G.729 (Annex B, SID 포함) / UDP — INVITE 가 IP 조각(fragment) 으로 나뉨
  3) G.711 A-law (PCMA) / SIP over TCP — 상대가 486 Busy 후 재시도 없이 종료 → 다음 통화는 정상
"""

from __future__ import annotations

import math
import random
import struct
import sys

from scapy.all import IP, TCP, UDP, Ether, Raw, fragment, wrpcap

A, B = "192.168.0.10", "192.168.0.20"
RATE = 8000


def _seg(val: int, ends: list[int]) -> int:
    for i, e in enumerate(ends):
        if val <= e:
            return i
    return len(ends)


def lin2ulaw(x: int) -> int:
    """ITU-T G.711 mu-law 인코더 (Sun g711.c 와 동일)."""
    x >>= 2
    mask = 0xFF
    if x < 0:
        x, mask = -x, 0x7F
    x = min(x, 8159) + (0x84 >> 2)
    seg = _seg(x, [0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])
    if seg >= 8:
        return 0x7F ^ mask
    return ((seg << 4) | ((x >> (seg + 1)) & 0x0F)) ^ mask


def lin2alaw(x: int) -> int:
    """ITU-T G.711 A-law 인코더 (Sun g711.c 와 동일)."""
    x >>= 3
    if x >= 0:
        mask = 0xD5
    else:
        mask, x = 0x55, -x - 1
    seg = _seg(x, [0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])
    if seg >= 8:
        return 0x7F ^ mask
    aval = seg << 4
    aval |= ((x >> 1) if seg < 2 else (x >> seg)) & 0x0F
    return aval ^ mask


def sine(freq: float, n: int, start: int, amp: int = 8000) -> list[int]:
    return [int(amp * math.sin(2 * math.pi * freq * (start + i) / RATE)) for i in range(n)]


def rtp(seq, ts, ssrc, pt, payload, marker=False) -> bytes:
    return struct.pack("!BBHII", 0x80, (0x80 if marker else 0) | pt, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc) + payload


def sip_msg(start, headers: dict, body: str = "") -> bytes:
    h = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    b = body.replace("\n", "\r\n")
    return f"{start}\r\n{h}Content-Length: {len(b.encode())}\r\n\r\n{b}".encode()


def sdp(ip, port, pts: list[int], extra: str) -> str:
    return (f"v=0\no=- 1 1 IN IP4 {ip}\ns=-\nc=IN IP4 {ip}\nt=0 0\n"
            f"m=audio {port} RTP/AVP {' '.join(map(str, pts))}\n{extra}a=sendrecv\n")


def dialog(call_id, ftag, ttag, from_user, to_user, cseq=1):
    base = {"Via": f"SIP/2.0/UDP {A}:5060;branch=z9hG4bK{call_id[:6]}",
            "From": f'"{from_user}" <sip:{from_user}@{A}>;tag={ftag}',
            "To": f"<sip:{to_user}@{B}>",
            "Call-ID": call_id, "CSeq": f"{cseq} INVITE"}
    to_tagged = dict(base, To=f"<sip:{to_user}@{B}>;tag={ttag}")
    return base, to_tagged


def udp(t, src, sport, dst, dport, data):
    p = Ether() / IP(src=src, dst=dst) / UDP(sport=sport, dport=dport) / Raw(data)
    p.time = t
    return p


def make(path: str) -> None:
    random.seed(1)
    pkts = []
    t0 = 1_760_000_000.0

    # ---------------- 통화 1: PCMU ----------------
    cid = "call-one-pcmu@192.168.0.10"
    req, resp = dialog(cid, "aaa1", "bbb1", "1001", "2002")
    inv_sdp = sdp(A, 40000, [0, 8, 18, 101], "a=rtpmap:0 PCMU/8000\na=rtpmap:18 G729/8000\na=rtpmap:101 telephone-event/8000\n")
    ok_sdp = sdp(B, 50000, [0, 101], "a=rtpmap:0 PCMU/8000\na=rtpmap:101 telephone-event/8000\n")
    t = t0
    pkts.append(udp(t, A, 5060, B, 5060, sip_msg(f"INVITE sip:2002@{B} SIP/2.0", dict(req, **{"Content-Type": "application/sdp"}), inv_sdp)))
    pkts.append(udp(t + 0.05, B, 5060, A, 5060, sip_msg("SIP/2.0 180 Ringing", resp)))
    pkts.append(udp(t + 1.0, B, 5060, A, 5060, sip_msg("SIP/2.0 200 OK", dict(resp, **{"Content-Type": "application/sdp"}), ok_sdp)))
    ack = dict(resp, CSeq="1 ACK")
    pkts.append(udp(t + 1.01, A, 5060, B, 5060, sip_msg(f"ACK sip:2002@{B} SIP/2.0", ack)))
    n_pkt = 150                                   # 3 초
    rtp_a, rtp_b = [], []
    for i in range(n_pkt):
        ta = t + 1.02 + i * 0.02
        pa = bytes(lin2ulaw(s) for s in sine(440, 160, i * 160))
        pb = bytes(lin2ulaw(s) for s in sine(880, 160, i * 160))
        if i != 50:                               # A→B 50번 패킷 손실
            rtp_a.append(udp(ta, A, 40000, B, 50000, rtp(1000 + i, 5000 + i * 160, 0x1111, 0, pa, i == 0)))
        rtp_b.append(udp(ta + 0.003, B, 50000, A, 40000, rtp(65500 + i, 4294960000 + i * 160, 0x2222, 0, pb, i == 0)))  # seq/ts wrap
    rtp_a[70], rtp_a[71] = rtp_a[71], rtp_a[70]   # 순서 뒤바뀜
    pkts += rtp_a + rtp_b
    tb = t + 1.02 + n_pkt * 0.02 + 0.1
    bye = dict(req, CSeq="2 BYE", To=f"<sip:2002@{B}>;tag=bbb1")
    pkts.append(udp(tb, A, 5060, B, 5060, sip_msg(f"BYE sip:2002@{B} SIP/2.0", bye)))
    pkts.append(udp(tb + 0.02, B, 5060, A, 5060, sip_msg("SIP/2.0 200 OK", dict(resp, CSeq="2 BYE"))))

    # ---------------- 통화 2: G.729, INVITE 조각남 ----------------
    cid = "call-two-g729@192.168.0.10"
    req, resp = dialog(cid, "aaa2", "bbb2", "1001", "3003")
    padding = "".join(f"a=x-filler{i}:{'z' * 60}\n" for i in range(25))     # SDP 를 키워서 MTU 초과
    inv_sdp = sdp(A, 40010, [18, 101], "a=rtpmap:18 G729/8000\na=fmtp:18 annexb=yes\na=rtpmap:101 telephone-event/8000\n" + padding)
    ok_sdp = sdp(B, 50010, [18, 101], "a=rtpmap:18 G729/8000\na=fmtp:18 annexb=yes\na=rtpmap:101 telephone-event/8000\n")
    t = t0 + 10
    inv = Ether() / IP(src=A, dst=B, id=4242) / UDP(sport=5060, dport=5060) / Raw(
        sip_msg(f"INVITE sip:3003@{B} SIP/2.0", dict(req, **{"Content-Type": "application/sdp"}), inv_sdp))
    frags = fragment(inv[IP], fragsize=1000)
    assert len(frags) >= 2
    for k, f in enumerate(reversed(frags)):      # 순서 뒤집어 보내도 재조립되는지
        e = Ether() / f
        e.time = t + k * 0.0001
        pkts.append(e)
    pkts.append(udp(t + 0.5, B, 5060, A, 5060, sip_msg("SIP/2.0 200 OK", dict(resp, **{"Content-Type": "application/sdp"}), ok_sdp)))
    pkts.append(udp(t + 0.51, A, 5060, B, 5060, sip_msg(f"ACK sip:3003@{B} SIP/2.0", dict(resp, CSeq="1 ACK"))))
    ts, seq = 0, 0
    for i in range(100):                         # 20 ms (2 프레임) 패킷
        tp = t + 0.52 + i * 0.02
        if 40 <= i < 60:
            pkts.append(udp(tp + 0.001, B, 50010, A, 40010, rtp(i, i * 160, 0x4444, 18, bytes(random.getrandbits(8) for _ in range(20)))))
            if i == 40:                          # 무음 구간 시작: SID 1개 후 전송 중단 (DTX)
                pkts.append(udp(tp, A, 40010, B, 50010, rtp(seq, ts, 0x3333, 18, bytes(random.getrandbits(8) for _ in range(2)))))
                seq += 1                         # DTX: seq 는 연속, timestamp 만 건너뜀
            ts += 160
            continue
        pay = bytes(random.getrandbits(8) for _ in range(20))
        pkts.append(udp(tp, A, 40010, B, 50010, rtp(seq, ts, 0x3333, 18, pay)))
        pkts.append(udp(tp + 0.001, B, 50010, A, 40010, rtp(i, i * 160, 0x4444, 18, bytes(random.getrandbits(8) for _ in range(20)))))
        seq += 1
        ts += 160
    # RTCP 같은 포트+1 로 섞여도 무시되는지
    pkts.append(udp(t + 1.0, A, 40011, B, 50011, bytes([0x80, 200, 0, 6]) + bytes(24)))
    tb = t + 0.52 + 100 * 0.02
    pkts.append(udp(tb, B, 5060, A, 5060, sip_msg(f"BYE sip:1001@{A} SIP/2.0",
                dict(resp, CSeq="1 BYE", From=resp["To"], To=req["From"]))))

    # ---------------- 통화 3: SIP over TCP, PCMA, 486 → 재통화 ----------------
    seq_a, seq_b = 1000, 5000

    def tcp_send(tt, src, dst, sport, dport, data, seqno, chunks=1):
        out = []
        size = math.ceil(len(data) / chunks)
        for k in range(chunks):
            part = data[k * size:(k + 1) * size]
            p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="PA", seq=seqno) / Raw(part)
            p.time = tt + k * 0.0001
            seqno += len(part)
            out.append(p)
        return out, seqno

    for attempt, (cid, code) in enumerate([("call-three-busy@tcp", 486), ("call-four-pcma@tcp", 200)]):
        req, resp = dialog(cid, f"aaa{3 + attempt}", f"bbb{3 + attempt}", "홍길동", "4004")
        req["Via"] = f"SIP/2.0/TCP {A}:5070;branch=z9hG4bK{attempt}"
        t = t0 + 20 + attempt * 5
        port = 40020 + attempt * 2
        body = sdp(A, port, [8], "a=rtpmap:8 PCMA/8000\n")
        p, seq_a = tcp_send(t, A, B, 5070, 5060, sip_msg(f"INVITE sip:4004@{B};transport=tcp SIP/2.0",
                            dict(req, **{"Content-Type": "application/sdp"}), body), seq_a, chunks=3)
        pkts += p
        if code == 486:
            p, seq_b = tcp_send(t + 0.3, B, A, 5060, 5070, sip_msg("SIP/2.0 486 Busy Here", resp), seq_b)
            pkts += p
            continue
        ok = sdp(B, 50020, [8], "a=rtpmap:8 PCMA/8000\n")
        # 180 과 200 이 TCP segment 하나에 같이 실려 옴
        both = sip_msg("SIP/2.0 180 Ringing", resp) + sip_msg("SIP/2.0 200 OK",
                                                              dict(resp, **{"Content-Type": "application/sdp"}), ok)
        p, seq_b = tcp_send(t + 0.3, B, A, 5060, 5070, both, seq_b)
        pkts += p
        for i in range(100):
            tp = t + 0.4 + i * 0.02
            pkts.append(udp(tp, A, port, B, 50020, rtp(i, i * 160, 0x5555, 8, bytes(lin2alaw(s) for s in sine(300, 160, i * 160)))))
            pkts.append(udp(tp + 0.002, B, 50020, A, port, rtp(i, i * 160, 0x6666, 8, bytes(lin2alaw(s) for s in sine(600, 160, i * 160)))))
        p, seq_a = tcp_send(t + 2.5, A, B, 5070, 5060, sip_msg(f"BYE sip:4004@{B} SIP/2.0",
                            dict(req, CSeq="2 BYE", To=resp["To"])), seq_a)
        pkts += p

    # REGISTER / OPTIONS 는 통화로 잡히면 안 됨
    pkts.append(udp(t0 + 40, A, 5060, B, 5060, sip_msg(f"OPTIONS sip:{B} SIP/2.0", {"Call-ID": "opt1", "CSeq": "1 OPTIONS",
                                                                                     "From": "<sip:a@x>;tag=1", "To": "<sip:b@y>"})))
    pkts.sort(key=lambda p: float(p.time))
    wrpcap(path, pkts)
    print(f"{path}: {len(pkts)} packets")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "sample.pcap")
