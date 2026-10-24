"""SIP 메시지 / SDP 파서 (RFC 3261, RFC 4566)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SIP_METHODS = {
    "INVITE", "ACK", "BYE", "CANCEL", "REGISTER", "OPTIONS", "PRACK", "SUBSCRIBE",
    "NOTIFY", "PUBLISH", "INFO", "REFER", "MESSAGE", "UPDATE",
}

# RFC 3261 7.3.3 compact header form
_COMPACT = {
    "i": "call-id", "f": "from", "t": "to", "m": "contact", "l": "content-length",
    "c": "content-type", "v": "via", "k": "supported", "s": "subject", "e": "content-encoding",
    "o": "event", "r": "refer-to", "b": "referred-by", "u": "allow-events", "x": "session-expires",
}


def looks_like_sip(data: bytes) -> bool:
    if data.startswith(b"SIP/2.0 "):
        return True
    sp = data.find(b" ", 0, 12)
    return sp > 0 and data[:sp].decode("ascii", "ignore") in SIP_METHODS


@dataclass
class SipMessage:
    start_line: str
    is_request: bool
    method: str | None = None          # 요청일 때
    request_uri: str = ""
    status_code: int | None = None     # 응답일 때
    reason: str = ""
    headers: dict[str, list[str]] = field(default_factory=dict)
    body: bytes = b""

    def header(self, name: str) -> str | None:
        v = self.headers.get(name.lower())
        return v[0] if v else None

    @property
    def call_id(self) -> str | None:
        return self.header("call-id")

    @property
    def cseq(self) -> tuple[int, str]:
        v = self.header("cseq") or ""
        parts = v.split()
        try:
            return int(parts[0]), parts[1].upper() if len(parts) > 1 else ""
        except (ValueError, IndexError):
            return 0, ""

    @property
    def from_tag(self) -> str | None:
        return header_param(self.header("from"), "tag")

    @property
    def to_tag(self) -> str | None:
        return header_param(self.header("to"), "tag")

    @property
    def content_type(self) -> str:
        return (self.header("content-type") or "").split(";")[0].strip().lower()


def _parse_head(head: str) -> SipMessage | None:
    lines = head.split("\r\n") if "\r\n" in head else head.split("\n")
    first = lines[0].strip()
    if first.startswith("SIP/2.0 "):
        parts = first.split(" ", 2)
        try:
            code = int(parts[1])
        except (IndexError, ValueError):
            return None
        msg = SipMessage(first, False, status_code=code, reason=parts[2] if len(parts) > 2 else "")
    else:
        parts = first.split(" ")
        if len(parts) < 3 or parts[0] not in SIP_METHODS or not parts[2].startswith("SIP/"):
            return None
        msg = SipMessage(first, True, method=parts[0], request_uri=parts[1])

    last = None
    for line in lines[1:]:
        if not line:
            continue
        if line[0] in " \t" and last:              # header 줄 이어짐 (folding)
            msg.headers[last][-1] += " " + line.strip()
            continue
        name, sep, value = line.partition(":")
        if not sep:
            continue
        name = name.strip().lower()
        name = _COMPACT.get(name, name)
        msg.headers.setdefault(name, []).append(value.strip())
        last = name
    return msg


def parse_sip(data: bytes) -> SipMessage | None:
    """UDP 데이터그램 하나 = SIP 메시지 하나."""
    msg, _ = parse_sip_stream(data, datagram=True)
    return msg


def parse_sip_stream(data: bytes, datagram: bool = False) -> tuple[SipMessage | None, int]:
    """버퍼 맨 앞의 SIP 메시지 하나를 파싱. (메시지, 소비한 바이트 수). 아직 덜 왔으면 (None, 0)."""
    sep = data.find(b"\r\n\r\n")
    sep_len = 4
    if sep < 0:
        sep = data.find(b"\n\n")
        sep_len = 2
    if sep < 0:
        if datagram:
            sep, sep_len = len(data), 0
        else:
            return None, 0
    head = data[:sep].decode("utf-8", "replace")
    msg = _parse_head(head)
    if msg is None:
        return None, -1
    body_start = sep + sep_len
    clen = msg.header("content-length")
    if clen is not None and clen.strip().isdigit():
        n = int(clen.strip())
        if not datagram and len(data) < body_start + n:
            return None, 0
        msg.body = data[body_start:body_start + n]
        return msg, body_start + n
    if datagram:
        msg.body = data[body_start:]
        return msg, len(data)
    return msg, body_start   # TCP 인데 Content-Length 없음 → body 없음으로 간주


# ---------------------------------------------------------------------------
# 헤더 도우미
# ---------------------------------------------------------------------------

def header_param(value: str | None, name: str) -> str | None:
    if not value:
        return None
    # <...> 안의 URI 파라미터는 제외하고 헤더 파라미터만 본다
    tail = value.rsplit(">", 1)[-1] if ">" in value else value
    m = re.search(r";\s*" + re.escape(name) + r"\s*=\s*([^;\s,]+)", tail, re.IGNORECASE)
    return m.group(1) if m else None


_URI_RE = re.compile(r"(?:sips?|tel):([^@;>\s]+)(?:@([^;>\s]+))?", re.IGNORECASE)


def display_party(value: str | None) -> str:
    """'"홍길동" <sip:1001@10.0.0.1>;tag=x' → '홍길동 <1001>'"""
    if not value:
        return ""
    name = ""
    if "<" in value:
        name = value.split("<", 1)[0].strip().strip('"').strip()
    m = _URI_RE.search(value)
    user = m.group(1) if m else value.split(";")[0].strip("<> ")
    return f"{name} <{user}>" if name and name != user else user


# ---------------------------------------------------------------------------
# SDP
# ---------------------------------------------------------------------------

@dataclass
class MediaDesc:
    media: str                       # audio / video / ...
    ip: str | None
    port: int
    payload_types: list[int]
    rtpmap: dict[int, str] = field(default_factory=dict)    # PT → 코덱 이름 (예: 18 → 'G729')
    fmtp: dict[int, str] = field(default_factory=dict)
    direction: str = "sendrecv"


def parse_sdp(body: bytes) -> list[MediaDesc]:
    text = body.decode("utf-8", "replace")
    session_ip = None
    medias: list[MediaDesc] = []
    cur: MediaDesc | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < 2 or line[1] != "=":
            continue
        k, v = line[0], line[2:].strip()
        if k == "c":
            parts = v.split()
            ip = parts[2].split("/")[0] if len(parts) >= 3 else None
            if cur is None:
                session_ip = ip
            else:
                cur.ip = ip
        elif k == "m":
            parts = v.split()
            if len(parts) < 3:
                cur = None
                continue
            try:
                port = int(parts[1].split("/")[0])
            except ValueError:
                port = 0
            pts = [int(p) for p in parts[3:] if p.isdigit()]
            cur = MediaDesc(parts[0].lower(), session_ip, port, pts)
            medias.append(cur)
        elif k == "a" and cur is not None:
            if v.startswith("rtpmap:"):
                pt_s, _, enc = v[7:].partition(" ")
                if pt_s.strip().isdigit():
                    cur.rtpmap[int(pt_s)] = enc.split("/")[0].strip()
            elif v.startswith("fmtp:"):
                pt_s, _, params = v[5:].partition(" ")
                if pt_s.strip().isdigit():
                    cur.fmtp[int(pt_s)] = params.strip()
            elif v in ("sendrecv", "sendonly", "recvonly", "inactive"):
                cur.direction = v
    # 세션 레벨 c= 가 m= 뒤에 오는 경우는 없지만, m= 에 c= 가 없으면 세션 값 사용
    for m in medias:
        if m.ip is None:
            m.ip = session_ip
    return medias
