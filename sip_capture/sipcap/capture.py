"""패킷 캡처 (Scapy + Npcap). 실시간 캡처, pcap/pcapng 파일 읽기, pcap 저장.

Windows 11 에서는 WinPcap 이 아니라 Npcap(https://npcap.com) 이 설치되어 있어야 한다.
"""

from __future__ import annotations

import struct
import time
from typing import Callable

from .calls import CallTracker

PacketTuple = tuple[float, str, int, str, int, str, bytes]


def _scapy():
    try:
        from scapy import all as sc   # noqa: F401  (layer 등록)
        from scapy.config import conf
        conf.verb = 0
        return sc
    except ImportError as e:
        raise RuntimeError("scapy 가 설치되어 있지 않습니다: pip install scapy") from e


class Ipv4Defrag:
    """조각난 IPv4 패킷 재조립 (SDP 가 큰 INVITE 는 1500바이트를 넘어 조각날 수 있음)."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.pending: dict[tuple, dict] = {}

    def add(self, t: float, src: str, dst: str, ident: int, proto: int, offset: int, more: bool,
            data: bytes) -> bytes | None:
        key = (src, dst, ident, proto)
        ent = self.pending.setdefault(key, {"t": t, "parts": {}, "total": None})
        ent["parts"][offset] = data
        if not more:
            ent["total"] = offset + len(data)
        if len(self.pending) > 256:                      # 오래된 조각 정리
            for k in [k for k, v in self.pending.items() if t - v["t"] > self.timeout]:
                del self.pending[k]
        if ent["total"] is None:
            return None
        buf, pos = bytearray(), 0
        for off in sorted(ent["parts"]):
            if off > pos:
                return None                              # 아직 빈 조각이 있음
            chunk = ent["parts"][off]
            buf += chunk[pos - off:]
            pos = max(pos, off + len(chunk))
        if pos < ent["total"]:
            return None
        del self.pending[key]
        return bytes(buf[:ent["total"]])


def packet_to_tuple(pkt, defrag: Ipv4Defrag | None = None) -> PacketTuple | None:
    """Scapy 패킷 → (시간, src_ip, sport, dst_ip, dport, 'UDP'|'TCP', payload)."""
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6

    t = float(pkt.time)
    if IP in pkt:
        ip = pkt[IP]
        mf = bool(int(ip.flags) & 0x1)                     # More Fragments flag
        if ip.frag or mf:
            if defrag is None or ip.proto != 17:
                return None
            raw = bytes(ip.payload)[:max(0, ip.len - ip.ihl * 4)]
            whole = defrag.add(t, ip.src, ip.dst, ip.id, ip.proto, ip.frag * 8, mf, raw)
            if whole is None or len(whole) < 8:
                return None
            sport, dport, ulen = struct.unpack("!HHH", whole[:6])
            return t, ip.src, sport, ip.dst, dport, "UDP", whole[8:ulen if ulen >= 8 else None]
        src, dst = ip.src, ip.dst
    elif IPv6 in pkt:
        src, dst = pkt[IPv6].src, pkt[IPv6].dst
    else:
        return None
    if UDP in pkt:
        u = pkt[UDP]
        return t, src, u.sport, dst, u.dport, "UDP", bytes(u.payload)
    if TCP in pkt:
        tc = pkt[TCP]
        return t, src, tc.sport, dst, tc.dport, "TCP", bytes(tc.payload)
    return None


def list_interfaces() -> list[tuple[str, str]]:
    """(캡처에 넘길 이름, 사람이 읽을 설명) 목록. Windows 에서는 'Wi-Fi', '이더넷' 같은 이름."""
    _scapy()
    from scapy.config import conf
    result = []
    for iface in conf.ifaces.values():
        desc = getattr(iface, "description", "") or iface.name
        ips = []
        for attr in ("ip", "ips"):
            v = getattr(iface, attr, None)
            if isinstance(v, str) and v:
                ips.append(v)
            elif isinstance(v, dict):
                ips += [a for a in v.get(4, [])]
        label = f"{iface.name}  -  {desc}" + (f"  [{', '.join(dict.fromkeys(ips))}]" if ips else "")
        result.append((iface.name, label))
    return result


def default_interface() -> str | None:
    _scapy()
    from scapy.config import conf
    try:
        return conf.iface.name if hasattr(conf.iface, "name") else str(conf.iface)
    except Exception:
        return None


DEFAULT_BPF = "udp or tcp port 5060"


class LiveCapture:
    """백그라운드 스레드에서 캡처. on_packet(tuple) 이 캡처 스레드에서 호출된다."""

    def __init__(self, iface: str | None, on_packet: Callable[[PacketTuple], None],
                 bpf: str | None = DEFAULT_BPF, pcap_path: str | None = None):
        self.iface = iface
        self.bpf = bpf or None
        self.on_packet = on_packet
        self.pcap_path = pcap_path
        self._sniffer = None
        self._writer = None
        self._defrag = Ipv4Defrag()
        self.captured = 0
        self.error: str | None = None

    def _cb(self, pkt):
        self.captured += 1
        if self._writer is not None:
            self._writer.write(pkt)
        try:
            tup = packet_to_tuple(pkt, self._defrag)
        except Exception:
            return
        if tup is not None:
            self.on_packet(tup)

    def _launch(self, sc, bpf: str | None) -> Exception | None:
        kw = dict(prn=self._cb, store=False)
        if self.iface:
            kw["iface"] = self.iface
        if bpf:
            kw["filter"] = bpf
        self._sniffer = sc.AsyncSniffer(**kw)
        self._sniffer.start()
        time.sleep(0.3)                     # 시작 직후 에러(권한/Npcap 없음/필터 오류) 확인
        exc = getattr(self._sniffer, "exception", None)
        if exc is None and not self._sniffer.running:
            exc = RuntimeError("캡처를 시작하지 못했습니다 (Npcap 설치/관리자 권한 확인)")
        if exc is not None:
            self.stop_sniffer()
        return exc

    def start(self) -> None:
        sc = _scapy()
        if self.pcap_path:
            self._writer = sc.PcapWriter(self.pcap_path, append=False, sync=False)
        exc = self._launch(sc, self.bpf)
        if exc is not None and self.bpf and "filter" in str(exc).lower():
            # 필터를 못 쓰는 환경(libpcap 없음, 문법 오류) → 필터 없이 캡처. SIP/RTP 는 내용으로 구분하므로 동작함
            self.warning = f"캡처 필터 사용 불가 → 필터 없이 캡처합니다 ({exc})"
            self.bpf = None
            exc = self._launch(sc, None)
        if exc is not None:
            self.error = str(exc)
            self._close_writer()
            raise RuntimeError(self.error)

    warning: str | None = None

    @property
    def running(self) -> bool:
        return bool(self._sniffer and self._sniffer.running and getattr(self._sniffer, "exception", None) is None)

    def stop_sniffer(self) -> None:
        if self._sniffer is not None and self._sniffer.running:
            try:
                self._sniffer.stop()
            except Exception:
                pass

    def stop(self) -> None:
        self.stop_sniffer()
        self._close_writer()

    def _close_writer(self):
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None


def read_pcap(path: str, tracker: CallTracker,
              progress: Callable[[int], None] | None = None) -> int:
    """pcap / pcapng 파일을 읽어 tracker 에 넣음. 읽은 패킷 수 반환."""
    sc = _scapy()
    defrag = Ipv4Defrag()
    n = 0
    with sc.PcapReader(path) as reader:          # pcapng 도 자동 인식
        for pkt in reader:
            n += 1
            try:
                tup = packet_to_tuple(pkt, defrag)
            except Exception:
                continue
            if tup is not None:
                tracker.process_packet(*tup)
            if progress and n % 5000 == 0:
                progress(n)
    return n
