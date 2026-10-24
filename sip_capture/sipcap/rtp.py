"""RTP 헤더 파서 (RFC 3550)."""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(slots=True)
class RtpPacket:
    marker: bool
    payload_type: int
    seq: int
    timestamp: int
    ssrc: int
    payload: bytes


def parse_rtp(data: bytes) -> RtpPacket | None:
    if len(data) < 12:
        return None
    b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", data[:12])
    if b0 >> 6 != 2:
        return None
    pt = b1 & 0x7F
    if 72 <= pt <= 76:            # RTCP (SR/RR/SDES/BYE/APP) 가 같은 포트로 온 경우
        return None
    off = 12 + 4 * (b0 & 0x0F)    # CSRC
    if b0 & 0x10:                 # header extension
        if len(data) < off + 4:
            return None
        ext_len = struct.unpack("!H", data[off + 2:off + 4])[0]
        off += 4 + 4 * ext_len
    end = len(data)
    if b0 & 0x20:                 # padding
        pad = data[-1]
        if pad == 0 or pad > end - off:
            return None
        end -= pad
    if off > end:
        return None
    return RtpPacket(bool(b1 & 0x80), pt, seq, ts, ssrc, data[off:end])


def looks_like_voice_rtp(pkt: RtpPacket, sport: int, dport: int) -> bool:
    """SIP 시그널링 없이 RTP 를 추측할 때(heuristic) 쓰는 조건: 음성 코덱 PT + 그럴듯한 크기."""
    if sport < 1024 or dport < 1024:
        return False
    n = len(pkt.payload)
    if pkt.payload_type in (0, 8):
        return 40 <= n <= 480 and n % 8 == 0
    if pkt.payload_type == 18:
        return 2 <= n <= 120 and n % 10 in (0, 2)
    return False
