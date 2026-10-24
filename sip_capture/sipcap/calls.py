"""SIP 통화(dialog) 추적 + SDP 로 RTP 스트림을 통화에 연결.

패킷 캡처 라이브러리와 독립적: process_packet() 에 (시간, IP/포트, 프로토콜, payload) 만 넘기면 된다.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime

from .codecs import STATIC_PAYLOAD_TYPES, normalize_codec
from .rtp import RtpPacket, looks_like_voice_rtp, parse_rtp
from .sip import SipMessage, display_party, looks_like_sip, parse_sdp, parse_sip, parse_sip_stream

CALLER, CALLEE = "caller", "callee"


@dataclass
class SipEvent:
    time: float
    src: str
    dst: str
    transport: str
    start_line: str
    raw: bytes


@dataclass
class Endpoint:
    """SDP 에 적힌 '여기로 RTP 를 보내 달라'는 주소."""
    call: "Call"
    side: str                     # 이 주소를 SDP 에 적은 쪽 (caller / callee)
    rtpmap: dict[int, str]


@dataclass
class RtpStream:
    src: tuple[str, int]
    dst: tuple[str, int]
    ssrc: int
    call: "Call | None" = None
    direction: str = "?"          # 'caller->callee' / 'callee->caller' / '?'
    rtpmap: dict[int, str] = field(default_factory=dict)
    # (도착 시각, 확장 seq, 확장 timestamp, PT, payload)
    packets: list[tuple[float, int, int, int, bytes]] = field(default_factory=list)
    first_time: float = 0.0
    last_time: float = 0.0
    lost: int = 0
    jitter: float = 0.0           # RTP timestamp 단위 (8 kHz → /8 하면 ms)
    _max_seq: int = -1
    _base_seq: int = 0
    _last_ts: int = 0
    _prev_transit: float | None = None
    _seen: set = field(default_factory=set)

    @property
    def key(self):
        return (self.src, self.dst, self.ssrc)

    def codec_of(self, pt: int) -> str:
        return normalize_codec(self.rtpmap.get(pt) or STATIC_PAYLOAD_TYPES.get(pt)) or f"PT{pt}"

    @property
    def codecs(self) -> list[str]:
        pts = OrderedDict.fromkeys(p[3] for p in self.packets)
        return [self.codec_of(pt) for pt in pts]

    @property
    def main_codec(self) -> str:
        from .codecs import NON_AUDIO
        counts: dict[str, int] = {}
        for p in self.packets:
            c = self.codec_of(p[3])
            if c not in NON_AUDIO:
                counts[c] = counts.get(c, 0) + 1
        return max(counts, key=counts.get) if counts else (self.codecs[0] if self.packets else "")

    @property
    def jitter_ms(self) -> float:
        return self.jitter / 8.0

    @property
    def expected(self) -> int:
        return (self._max_seq - self._base_seq + 1) if self._max_seq >= 0 else 0

    def add(self, t: float, pkt: RtpPacket) -> None:
        if self._max_seq < 0:
            ext_seq, ext_ts = pkt.seq, pkt.timestamp
            self._base_seq, self._max_seq, self._last_ts = ext_seq, ext_seq, ext_ts
            self.first_time = t
        else:
            d = (pkt.seq - (self._max_seq & 0xFFFF)) & 0xFFFF
            if d >= 0x8000:
                d -= 0x10000
            ext_seq = self._max_seq + d
            dt = (pkt.timestamp - (self._last_ts & 0xFFFFFFFF)) & 0xFFFFFFFF
            if dt >= 0x80000000:
                dt -= 0x100000000
            ext_ts = self._last_ts + dt
            if ext_seq > self._max_seq:
                self._max_seq = ext_seq
                self._last_ts = ext_ts
        if ext_seq in self._seen:          # 중복 패킷
            return
        self._seen.add(ext_seq)
        self.last_time = max(self.last_time, t)
        # RFC 3550 A.8 interarrival jitter
        transit = t * 8000.0 - ext_ts
        if self._prev_transit is not None:
            self.jitter += (abs(transit - self._prev_transit) - self.jitter) / 16.0
        self._prev_transit = transit
        self.packets.append((t, ext_seq, ext_ts, pkt.payload_type, pkt.payload))
        self.lost = max(0, self.expected - len(self.packets))


@dataclass
class Call:
    call_id: str
    from_: str
    to: str
    caller_tag: str | None
    start_time: float
    state: str = "통화 시도"
    connect_time: float | None = None
    end_time: float | None = None
    last_time: float = 0.0
    events: list[SipEvent] = field(default_factory=list)
    streams: list[RtpStream] = field(default_factory=list)
    caller_addr: str = ""
    callee_addr: str = ""
    index: int = 0

    @property
    def duration(self) -> float:
        start = self.connect_time or self.start_time
        end = self.end_time or self.last_time
        return max(0.0, end - start) if self.connect_time else 0.0

    @property
    def codec(self) -> str:
        names = OrderedDict.fromkeys(s.main_codec for s in self.streams if s.packets)
        return ", ".join(n for n in names if n)

    @property
    def start_str(self) -> str:
        return datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S")


class CallTracker:
    """패킷을 받아 통화/스트림 상태를 갱신. GUI 스레드와 공유할 때는 .lock 을 잡고 읽는다."""

    def __init__(self, rtp_heuristic: bool = False):
        self.rtp_heuristic = rtp_heuristic
        self.calls: "OrderedDict[str, Call]" = OrderedDict()
        self.streams: dict[tuple, RtpStream] = {}
        self.orphan_streams: list[RtpStream] = []
        self.endpoints: dict[tuple[str, int], Endpoint] = {}
        self.port_index: dict[int, Endpoint] = {}
        self.sip_count = 0
        self.rtp_count = 0
        self.packet_count = 0
        self.version = 0                    # 변경될 때마다 증가 (GUI 갱신 판단용)
        self.lock = threading.RLock()
        self._tcp_buffers: dict[tuple, bytearray] = {}

    # ------------------------------------------------------------------
    def process_packet(self, t: float, src_ip: str, sport: int, dst_ip: str, dport: int,
                       transport: str, payload: bytes) -> None:
        with self.lock:
            self.packet_count += 1
            if not payload:
                return
            if transport == "TCP":
                self._process_tcp(t, src_ip, sport, dst_ip, dport, payload)
                return
            if looks_like_sip(payload):
                msg = parse_sip(payload)
                if msg:
                    self._handle_sip(t, src_ip, sport, dst_ip, dport, "UDP", msg, payload)
                return
            self._handle_udp(t, src_ip, sport, dst_ip, dport, payload)

    def _process_tcp(self, t, src_ip, sport, dst_ip, dport, payload):
        key = (src_ip, sport, dst_ip, dport)
        buf = self._tcp_buffers.get(key)
        if buf is None:
            if not looks_like_sip(payload):
                return
            buf = self._tcp_buffers[key] = bytearray()
        buf += payload
        while buf:
            msg, used = parse_sip_stream(bytes(buf))
            if used == 0:                       # 다음 segment 기다림
                if len(buf) > 256 * 1024:
                    del self._tcp_buffers[key]
                return
            if used < 0 or msg is None:         # 동기 깨짐 → 버림
                del self._tcp_buffers[key]
                return
            raw = bytes(buf[:used])
            del buf[:used]
            self._handle_sip(t, src_ip, sport, dst_ip, dport, "TCP", msg, raw)
            while buf[:2] == b"\r\n":           # keep-alive CRLF
                del buf[:2]
        self._tcp_buffers.pop(key, None)

    # ------------------------------------------------------------------
    def _handle_sip(self, t, src_ip, sport, dst_ip, dport, transport, msg: SipMessage, raw: bytes):
        self.sip_count += 1
        call_id = msg.call_id
        if not call_id:
            return
        cseq_no, cseq_method = msg.cseq
        call = self.calls.get(call_id)
        if call is None:
            if not (msg.is_request and msg.method == "INVITE"):
                return                          # REGISTER/OPTIONS 등 통화 아닌 대화는 무시
            call = Call(call_id, display_party(msg.header("from")), display_party(msg.header("to")),
                        msg.from_tag, t, caller_addr=f"{src_ip}:{sport}", callee_addr=f"{dst_ip}:{dport}",
                        index=len(self.calls) + 1)
            self.calls[call_id] = call
        call.last_time = max(call.last_time, t)
        call.events.append(SipEvent(t, f"{src_ip}:{sport}", f"{dst_ip}:{dport}", transport, msg.start_line, raw))

        # 상태 갱신
        if msg.is_request:
            if msg.method == "BYE":
                if call.end_time is None:
                    call.state = "종료"
                    call.end_time = t
            elif msg.method == "CANCEL" and call.connect_time is None:
                call.state = "취소"
                call.end_time = t
        elif cseq_method == "INVITE" and msg.status_code:
            code = msg.status_code
            if 180 <= code < 200 and call.connect_time is None:
                call.state = "벨 울림"
            elif 200 <= code < 300 and call.connect_time is None:
                call.state = "연결됨"
                call.connect_time = t
            elif code >= 300 and call.connect_time is None and call.state != "취소":
                call.state = {486: "Busy(486)", 487: "취소(487)", 480: "부재(480)", 404: "없는 번호(404)",
                              603: "거절(603)"}.get(code, f"실패 {code}")
                call.end_time = t

        # SDP → RTP 주소 등록
        if msg.body and msg.content_type in ("application/sdp", ""):
            side = self._sender_side(call, msg)
            for m in parse_sdp(msg.body):
                if m.media != "audio" or not m.port or m.ip == "0.0.0.0":   # 0.0.0.0 = 보류(hold)
                    continue
                ip = m.ip or src_ip
                ep = Endpoint(call, side, dict(m.rtpmap))
                self.endpoints[(ip, m.port)] = ep
                self.port_index[m.port] = ep
        self.version += 1

    @staticmethod
    def _sender_side(call: Call, msg: SipMessage) -> str:
        """From tag 로 이 메시지를 보낸 쪽을 판단 (프록시를 거쳐도 동작)."""
        from_is_caller = (msg.from_tag == call.caller_tag) if call.caller_tag else True
        if msg.is_request:
            return CALLER if from_is_caller else CALLEE
        return CALLEE if from_is_caller else CALLER

    # ------------------------------------------------------------------
    def _lookup_endpoint(self, ip: str, port: int) -> Endpoint | None:
        return self.endpoints.get((ip, port))

    def _handle_udp(self, t, src_ip, sport, dst_ip, dport, payload):
        key_src, key_dst = (src_ip, sport), (dst_ip, dport)
        # 이미 아는 스트림인지 SSRC 까지 보고 판단
        pkt = parse_rtp(payload)
        if pkt is None:
            return
        skey = (key_src, key_dst, pkt.ssrc)
        stream = self.streams.get(skey)
        if stream is None:
            ep, direction = None, "?"
            dst_ep = self._lookup_endpoint(dst_ip, dport)
            src_ep = self._lookup_endpoint(src_ip, sport)
            if dst_ep:                                   # SDP 에 적힌 주소로 들어가는 RTP
                ep = dst_ep
                direction = "callee->caller" if dst_ep.side == CALLER else "caller->callee"
            elif src_ep:                                 # symmetric RTP: SDP 주소에서 나가는 RTP
                ep = src_ep
                direction = "caller->callee" if src_ep.side == CALLER else "callee->caller"
            elif dport in self.port_index or sport in self.port_index:   # NAT: 포트만 일치
                ep = self.port_index.get(dport)
                if ep is not None:
                    direction = "callee->caller" if ep.side == CALLER else "caller->callee"
                else:
                    ep = self.port_index[sport]
                    direction = "caller->callee" if ep.side == CALLER else "callee->caller"
            elif not (self.rtp_heuristic and looks_like_voice_rtp(pkt, sport, dport)):
                return
            stream = RtpStream(key_src, key_dst, pkt.ssrc)
            if ep is not None:
                stream.call = ep.call
                stream.direction = direction
                stream.rtpmap = dict(ep.rtpmap)
                ep.call.streams.append(stream)
            else:
                self.orphan_streams.append(stream)
            self.streams[skey] = stream
        stream.add(t, pkt)
        if stream.call is not None:
            stream.call.last_time = max(stream.call.last_time, t)
        self.rtp_count += 1
        self.version += 1

    # ------------------------------------------------------------------
    def call_list(self) -> list[Call]:
        with self.lock:
            return list(self.calls.values())
