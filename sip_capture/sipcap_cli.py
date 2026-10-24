"""SIP 캡처 명령줄 도구.

  python sipcap_cli.py interfaces                       # 캡처 가능한 네트워크 장치 목록
  python sipcap_cli.py live -i "Wi-Fi" -o out           # 실시간 캡처 (Ctrl+C 로 종료)
  python sipcap_cli.py live -i "이더넷" -t 600 -o out   # 10분 캡처
  python sipcap_cli.py read call.pcap -o out            # 저장된 pcap/pcapng 분석

결과: out/calls.csv (통화 목록), out/<통화>.wav (왼쪽=발신, 오른쪽=수신), out/<통화>.sip.txt (SIP 원문)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from sipcap import __version__
from sipcap.calls import CallTracker
from sipcap.codecs import g729_decoder_status
from sipcap.export import export_all


def _print_calls(tracker: CallTracker) -> None:
    calls = tracker.call_list()
    print(f"\nSIP {tracker.sip_count}개, RTP {tracker.rtp_count}개, 통화 {len(calls)}건")
    if not calls:
        return
    print(f"{'No':>3} {'시작':19} {'발신':22} {'수신':22} {'상태':10} {'시간(s)':>7} {'코덱':12} {'손실':>5}")
    for c in calls:
        print(f"{c.index:>3} {c.start_str:19} {c.from_[:22]:22} {c.to[:22]:22} {c.state[:10]:10} "
              f"{c.duration:7.1f} {c.codec[:12]:12} {sum(s.lost for s in c.streams):5}")
    if tracker.orphan_streams:
        print(f"(SIP 없이 감지된 RTP 스트림 {len(tracker.orphan_streams)}개)")


def _export(tracker: CallTracker, out: Path, per_stream: bool) -> None:
    print()
    export_all(tracker, out, per_stream)


def cmd_interfaces(_args) -> int:
    from sipcap.capture import default_interface, list_interfaces
    d = default_interface()
    for name, label in list_interfaces():
        print(("* " if name == d else "  ") + label)
    print("\n-i 에는 맨 앞의 이름을 그대로 넣으세요 (공백이 있으면 따옴표).")
    return 0


def cmd_live(args) -> int:
    from sipcap.capture import LiveCapture
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tracker = CallTracker(rtp_heuristic=args.heuristic)
    pcap = None if args.no_pcap else str(out / f"capture_{datetime.now():%Y%m%d_%H%M%S}.pcap")
    cap = LiveCapture(args.iface, lambda tup: tracker.process_packet(*tup), args.filter, pcap)
    try:
        cap.start()
    except RuntimeError as e:
        print("캡처 시작 실패:", e)
        print("→ Npcap 설치 여부, 장치 이름(-i), 필터 문법을 확인하세요. 필요하면 관리자 권한으로 실행.")
        return 1
    if cap.warning:
        print("경고:", cap.warning)
    print(f"캡처 중... 장치={args.iface or '기본'} 필터='{cap.bpf or '없음'}'  (Ctrl+C 로 종료)")
    if pcap:
        print(f"pcap 저장: {pcap}")
    start = time.time()
    last_calls = -1
    try:
        while cap.running and (not args.duration or time.time() - start < args.duration):
            time.sleep(1)
            with tracker.lock:
                n = len(tracker.calls)
                msg = f"\r패킷 {cap.captured}  SIP {tracker.sip_count}  RTP {tracker.rtp_count}  통화 {n}   "
            print(msg, end="", flush=True)
            if n != last_calls and n:
                c = tracker.call_list()[-1]
                print(f"\n  새 통화: {c.from_} -> {c.to}")
                last_calls = n
    except KeyboardInterrupt:
        pass
    cap.stop()
    _print_calls(tracker)
    _export(tracker, out, args.per_stream)
    return 0


def cmd_read(args) -> int:
    from sipcap.capture import read_pcap
    tracker = CallTracker(rtp_heuristic=args.heuristic)
    n = read_pcap(args.pcap, tracker, progress=lambda k: print(f"\r{k} 패킷...", end="", flush=True))
    print(f"\r{n} 패킷 읽음        ")
    _print_calls(tracker)
    _export(tracker, Path(args.out), args.per_stream)
    return 0


def main(argv=None) -> int:
    from sipcap.capture import DEFAULT_BPF
    p = argparse.ArgumentParser(description=f"SIP/RTP 캡처 + G.711/G.729 음성 복원 v{__version__}")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("interfaces", help="네트워크 장치 목록").set_defaults(func=cmd_interfaces)

    def common(sp):
        sp.add_argument("-o", "--out", default="sipcap_out", help="결과 폴더 (기본 sipcap_out)")
        sp.add_argument("--heuristic", action="store_true",
                        help="SIP 없이도 RTP(PCMU/PCMA/G729) 로 보이는 UDP 를 스트림으로 잡음")
        sp.add_argument("--per-stream", action="store_true", help="방향별 mono WAV 도 저장")

    lv = sub.add_parser("live", help="실시간 캡처")
    lv.add_argument("-i", "--iface", help="장치 이름 (interfaces 명령으로 확인)")
    lv.add_argument("-t", "--duration", type=int, default=0, help="캡처 시간(초), 0=Ctrl+C 까지")
    lv.add_argument("-f", "--filter", default=DEFAULT_BPF, help=f"BPF 필터 (기본 '{DEFAULT_BPF}')")
    lv.add_argument("--no-pcap", action="store_true", help="pcap 파일 저장 안 함")
    common(lv)
    lv.set_defaults(func=cmd_live)

    rd = sub.add_parser("read", help="pcap/pcapng 파일 분석")
    rd.add_argument("pcap")
    common(rd)
    rd.set_defaults(func=cmd_read)

    args = p.parse_args(argv)
    print(g729_decoder_status())
    return args.func(args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # Windows 콘솔 한글
    sys.exit(main())
