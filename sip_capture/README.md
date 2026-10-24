# SIP Capture — Windows 11용 SIP/RTP 캡처 + G.711 / G.729 음성 복원

네트워크에서 **SIP 통화(시그널링)** 와 **RTP 음성**을 잡아서 통화 목록을 보여 주고,
통화 음성을 **WAV 파일**로 복원/재생하는 프로그램입니다. (GUI + 명령줄 둘 다 제공)

| 지원 | 내용 |
|---|---|
| 코덱 | **G.711** μ-law(PCMU, PT 0) / A-law(PCMA, PT 8), **G.729 / G.729A / G.729B**(PT 18, SID 포함), 동적 PT 는 SDP `a=rtpmap` 으로 인식 |
| SIP | UDP / TCP, compact header(`i:`, `f:` …), 조각난(fragment) IP 패킷 재조립, re-INVITE |
| 통화 상태 | 통화 시도 → 벨 울림 → 연결됨 → 종료 / 취소 / Busy(486) / 거절(603) … |
| 통계 | 스트림별 패킷 수, 손실(개수·%), 지터(RFC 3550), 길이 |
| 저장 | 통화별 **스테레오 WAV** (왼쪽 = 발신자, 오른쪽 = 수신자), 방향별 mono WAV, `calls.csv`, SIP 원문 `.sip.txt`, 원본 `pcap` |
| 입력 | 실시간 캡처(Npcap) 또는 Wireshark 등으로 저장한 `.pcap` / `.pcapng` |

> **Win10용 프로그램이 Win11에서 안 되는 이유**: 예전 프로그램은 대부분 **WinPcap** 을 씁니다.
> WinPcap 은 2013년에 개발이 끝나서 Windows 11 에서는 제대로 동작하지 않습니다.
> 이 프로그램은 그 후속인 **Npcap** 을 사용합니다.

---

## 1. 설치 (처음 한 번)

1. **Python 3.10 이상** — https://www.python.org/downloads/windows/
   설치 첫 화면에서 **"Add python.exe to PATH"** 체크.
2. **Npcap** — https://npcap.com/#download
   설치 옵션에서 **"Install Npcap in WinPcap API-compatible Mode"** 체크.
   (기존 WinPcap 이 깔려 있으면 먼저 제거)
3. 이 폴더에서 **`install.bat`** 더블클릭 (= `pip install -r requirements.txt`, scapy 설치).
4. **G.729 를 쓰려면 디코더 하나 추가** (G.711 은 없어도 됨)
   - 쉬운 방법: **ffmpeg** — https://www.gyan.dev/ffmpeg/builds/ 에서 `ffmpeg-release-essentials.zip` 을 받아
     압축 안의 `bin\ffmpeg.exe` 를 **이 폴더(또는 `bin\` 폴더)** 에 복사하거나 PATH 에 추가.
   - 선택: **bcg729.dll** ([bcg729](https://github.com/BelledonneCommunications/bcg729) 1.x 를 CMake 로 빌드)
     을 같은 폴더에 두면 이것을 먼저 씁니다. G.729B 무음 구간의 comfort noise 까지 재현됩니다
     (ffmpeg 는 무음 구간을 그냥 무음으로 둠).
   - 프로그램 아래쪽 상태줄에 `G.729 디코더: ffmpeg (...)` 처럼 표시되면 준비 완료.

## 2. 사용법 — GUI

`run_gui.bat` 더블클릭 (또는 `python sipcap_gui.py`).

1. **네트워크 장치** 선택 (Wi-Fi / 이더넷 등, IP 주소가 같이 표시됨)
2. **▶ 캡처 시작** → 전화를 걸고 받으면 통화 목록에 실시간으로 나타남 → **■ 중지**
3. 통화를 클릭하면 아래에 **RTP 스트림**(손실/지터) 과 **SIP 메시지**(원문) 가 보임
4. 통화 **더블클릭 = 재생**, 오른쪽 클릭 = WAV 저장 / SIP 원문 저장
5. **전체 내보내기** = 모든 통화를 한 폴더에 (WAV + CSV + SIP 원문)
6. **pcap 열기** = Wireshark 등으로 이미 저장한 파일 분석 (Npcap 없어도 됨)

- 결과/자동 저장 pcap 기본 위치: `C:\Users\<사용자>\SIPCapture\`
- 기본 캡처 필터 `udp or tcp port 5060` — SIP 서버가 다른 포트(예: 5080)를 쓰면 `udp or tcp port 5080` 처럼 수정
- **SIP 없이 RTP 추측**: SIP 를 못 잡은 경우(캡처를 통화 도중 시작 등)에도 PCMU/PCMA/G.729 로 보이는 UDP 를 스트림으로 잡음
- "장치 목록이 비어 있음 / 캡처 시작 실패" → Npcap 설치 확인, 그래도 안 되면 **관리자 권한으로 실행**

## 3. 사용법 — 명령줄

```bat
python sipcap_cli.py interfaces                         :: 장치 이름 확인
python sipcap_cli.py live -i "Wi-Fi" -o result          :: 실시간 캡처, Ctrl+C 로 종료
python sipcap_cli.py live -i "이더넷" -t 600 -o result  :: 10분간 캡처
python sipcap_cli.py read call.pcapng -o result --per-stream
```

| 옵션 | 의미 |
|---|---|
| `-o 폴더` | 결과 폴더 (기본 `sipcap_out`) |
| `-f "필터"` | BPF 캡처 필터 (기본 `udp or tcp port 5060`) |
| `--heuristic` | SIP 없이도 RTP 추측 |
| `--per-stream` | 방향별 mono WAV 도 저장 |
| `--no-pcap` | 실시간 캡처 시 pcap 저장 안 함 |

결과 폴더:
```
result/
├── calls.csv                                         # 통화 목록 (엑셀에서 바로 열림)
├── capture_20261024_101500.pcap                      # 원본 패킷 (Wireshark 로 열 수 있음)
├── 20261024_101530_1001_to_2002_a84f3c2e91b0.wav     # 왼쪽=발신, 오른쪽=수신
└── 20261024_101530_1001_to_2002_a84f3c2e91b0.sip.txt # SIP 메시지 원문
```

## 4. exe 로 만들기 (Python 없는 PC 에 배포)

`build_exe.bat` 실행 → `dist\SIPCapture.exe`(GUI), `dist\sipcap.exe`(CLI) 생성.
G.729 를 쓰려면 `ffmpeg.exe` 를 exe 옆에 같이 두세요. 배포받는 PC 에도 **Npcap 은 따로 설치**해야 합니다.

## 5. 어디서 캡처해야 하나요?

- **이 PC 의 소프트폰**(MicroSIP, Zoiper 등) 통화 → 이 PC 에서 바로 캡처하면 됨
- **책상 위 IP 전화기** 통화 → 전화기 트래픽이 이 PC 를 지나가지 않으므로,
  스위치의 **포트 미러링(SPAN)** 으로 전화기 포트를 PC 포트에 복사하거나, 전화기 → PC 브리지(PC 포트) 구성 필요
- 같은 PC 안에서 오가는 트래픽(예: 같은 PC 의 PBX ↔ 소프트폰)은 장치 목록의 **"Adapter for loopback traffic capture"**(Npcap Loopback) 를 선택

## 6. 한계

- **암호화된 통화는 복원 불가**: SIP over TLS(5061), SRTP. (전화기/PBX 설정에서 암호화를 끄면 가능)
- G.711 / G.729 외 코덱(G.722, Opus, AMR 등)은 목록·통계는 보이지만 WAV 로 복원하지 않음
- TCP SIP 는 순서대로 도착한 segment 만 재조립 (재전송/순서 뒤바뀜이 많으면 일부 메시지 누락 가능)
- 메모리에 RTP 를 들고 있으므로 수십 시간 연속 캡처는 pcap 으로 저장 후 나눠서 분석 권장

> ⚠️ 통화 내용 캡처·녹음은 **본인이 참여한 통화** 또는 **관리 권한이 있는 망의 장애 분석** 용도로만 사용하세요.
> 타인 간의 통화를 동의 없이 녹음하는 것은 통신비밀보호법 위반이 될 수 있습니다.

## 7. 동작 확인 (전화기 없이)

```bat
python tools\make_sample_pcap.py sample.pcap      :: 가짜 통화 4건 (PCMU / G.729+조각난 INVITE / TCP Busy / TCP PCMA)
python sipcap_cli.py read sample.pcap -o test_out
python -m pip install pytest && python -m pytest tests
```

## 파일 구성

```
sip_capture/
├── sipcap_gui.py          # GUI (tkinter)
├── sipcap_cli.py          # 명령줄
├── sipcap/
│   ├── capture.py         # Npcap 실시간 캡처, pcap 읽기/쓰기, IP fragment 재조립
│   ├── sip.py             # SIP / SDP 파서
│   ├── rtp.py             # RTP 헤더 파서
│   ├── calls.py           # 통화 추적, SDP 로 RTP ↔ 통화 연결, 손실/지터 통계
│   ├── codecs.py          # G.711 디코더, G.729 (bcg729 / ffmpeg)
│   └── export.py          # WAV / CSV / SIP 원문 저장
├── tools/make_sample_pcap.py
├── tests/test_sipcap.py
├── install.bat  run_gui.bat  build_exe.bat
└── requirements.txt
```
