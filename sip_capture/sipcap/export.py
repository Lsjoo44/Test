"""RTP 스트림 → PCM 복원 / WAV · CSV 저장."""

from __future__ import annotations

import csv
import re
import sys
import wave
from array import array
from datetime import datetime
from pathlib import Path

from .calls import Call, RtpStream
from .codecs import (NON_AUDIO, SAMPLE_RATE, decode_pcma, decode_pcmu,
                     get_g729_decoder, split_g729_payload)

MAX_SECONDS = 4 * 3600          # 비정상 timestamp 점프 방어


class DecoderMissing(RuntimeError):
    pass


def stream_to_pcm(stream: RtpStream, g729=None) -> array:
    """RTP timestamp 위치에 맞춰 디코딩한 샘플을 배치 (손실 구간은 무음). int16 array 반환."""
    pkts = sorted(stream.packets, key=lambda p: p[1])
    if not pkts:
        return array("h")
    base_ts = pkts[0][2]
    out = array("h")

    g729_pkts = [p for p in pkts if stream.codec_of(p[3]) == "G729"]
    g729_pcm: dict[int, list[bytes]] = {}
    if g729_pkts:
        dec = g729 or get_g729_decoder()
        if dec is None:
            raise DecoderMissing("G.729 디코더가 없습니다. bcg729.dll 또는 ffmpeg.exe 를 프로그램 폴더/PATH 에 두세요.")
        frames, owners = [], []
        for p in g729_pkts:
            fs = split_g729_payload(p[4])
            frames += fs
            owners += [p[1]] * len(fs)
        decoded = dec.decode(frames)
        for seq, pcm in zip(owners, decoded):
            g729_pcm.setdefault(seq, []).append(pcm)

    for _, seq, ts, pt, payload in pkts:
        codec = stream.codec_of(pt)
        if codec in NON_AUDIO:
            continue
        if codec == "PCMU":
            pcm = decode_pcmu(payload)
        elif codec == "PCMA":
            pcm = decode_pcma(payload)
        elif codec == "G729":
            pcm = b"".join(g729_pcm.get(seq, []))
        else:
            continue
        offset = ts - base_ts
        if offset < 0 or offset > MAX_SECONDS * SAMPLE_RATE:
            continue
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        end = offset + len(samples)
        if len(out) < end:
            out.extend(array("h", [0]) * (end - len(out)))
        out[offset:end] = samples
    return out


def _mix_into(dst: array, src: array, offset: int) -> None:
    """dst[offset:] 에 src 를 더함. 겹치지 않는 부분은 그대로 복사 (보통 방향당 스트림 1개라 빠름)."""
    if len(dst) < offset:
        dst.extend(array("h", [0]) * (offset - len(dst)))
    overlap = min(len(src), len(dst) - offset)
    for i in range(overlap):
        s = dst[offset + i] + src[i]
        dst[offset + i] = 32767 if s > 32767 else (-32768 if s < -32768 else s)
    dst.extend(src[overlap:])


def call_to_stereo(call: Call, g729=None) -> tuple[array, array]:
    """왼쪽 = 발신자(caller) 음성, 오른쪽 = 수신자(callee) 음성. 스트림 시작 시각 기준으로 정렬."""
    streams = [s for s in call.streams if s.packets]
    if not streams:
        return array("h"), array("h")
    t0 = min(s.first_time for s in streams)
    left, right = array("h"), array("h")
    for s in streams:
        pcm = stream_to_pcm(s, g729)
        if not pcm:
            continue
        offset = int(round((s.first_time - t0) * SAMPLE_RATE))
        if s.direction == "caller->callee":
            _mix_into(left, pcm, offset)
        elif s.direction == "callee->caller":
            _mix_into(right, pcm, offset)
        else:
            _mix_into(left if not left else right, pcm, offset)
    n = max(len(left), len(right))
    left.extend(array("h", [0]) * (n - len(left)))
    right.extend(array("h", [0]) * (n - len(right)))
    return left, right


def write_wav(path: str | Path, channels: list[array]) -> None:
    n = max((len(c) for c in channels), default=0)
    if len(channels) == 1:
        data = channels[0]
    else:
        data = array("h", [0]) * (n * len(channels))
        for ci, c in enumerate(channels):
            data[ci::len(channels)] = c + array("h", [0]) * (n - len(c))
    if sys.byteorder != "little":
        data = array("h", data)
        data.byteswap()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(len(channels))
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data.tobytes())


def save_call_wav(call: Call, path: str | Path, g729=None) -> Path:
    left, right = call_to_stereo(call, g729)
    write_wav(path, [left, right])
    return Path(path)


def save_stream_wav(stream: RtpStream, path: str | Path, g729=None) -> Path:
    write_wav(path, [stream_to_pcm(stream, g729)])
    return Path(path)


_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]+')


def safe_name(text: str, limit: int = 40) -> str:
    """Windows 파일 이름에 못 쓰는 문자 제거."""
    s = _BAD.sub("_", text).strip("._")
    return (s[:limit] or "unknown")


def call_filename(call: Call) -> str:
    ts = datetime.fromtimestamp(call.start_time).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{safe_name(call.from_)}_to_{safe_name(call.to)}_{safe_name(call.call_id, 12)}.wav"


def write_calls_csv(calls: list[Call], path: str | Path) -> None:
    # utf-8-sig: 엑셀에서 한글이 안 깨지게
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "start", "from", "to", "state", "duration_s", "codec", "rtp_streams",
                    "rtp_packets", "lost", "max_jitter_ms", "caller", "callee", "call_id"])
        for c in calls:
            w.writerow([c.index, c.start_str, c.from_, c.to, c.state, f"{c.duration:.1f}", c.codec,
                        len(c.streams), sum(len(s.packets) for s in c.streams),
                        sum(s.lost for s in c.streams),
                        f"{max((s.jitter_ms for s in c.streams), default=0):.2f}",
                        c.caller_addr, c.callee_addr, c.call_id])


def write_sip_log(call: Call, path: str | Path) -> None:
    """통화의 SIP 메시지 원문을 시간순 텍스트로 저장."""
    with open(path, "w", encoding="utf-8") as f:
        for e in call.events:
            t = datetime.fromtimestamp(e.time).strftime("%H:%M:%S.%f")[:-3]
            f.write(f"===== {t}  {e.src} -> {e.dst}  [{e.transport}]\n")
            f.write(e.raw.decode("utf-8", "replace").replace("\r\n", "\n"))
            f.write("\n\n")


def export_all(tracker, out: str | Path, per_stream: bool = False, log=print) -> Path:
    """모든 통화를 out 폴더에: calls.csv + 통화별 stereo WAV + SIP 원문 (+ 방향별 mono WAV)."""
    from .codecs import is_supported
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    calls = tracker.call_list()
    write_calls_csv(calls, out / "calls.csv")
    log(f"저장 폴더: {out.resolve()}")
    log("  calls.csv")
    with tracker.lock:
        for c in calls:
            name = call_filename(c)
            write_sip_log(c, out / name.replace(".wav", ".sip.txt"))
            streams = [s for s in c.streams if s.packets]
            if not streams:
                continue
            if not any(is_supported(s.main_codec) for s in streams):
                log(f"  [건너뜀] 통화 {c.index}: 지원하지 않는 코덱 {c.codec}")
                continue
            try:
                save_call_wav(c, out / name)
                log(f"  {name}")
                if per_stream:
                    for i, s in enumerate(streams, 1):
                        p = out / name.replace(".wav", f".stream{i}_{s.direction.replace('->', '_to_')}.wav")
                        save_stream_wav(s, p)
                        log(f"  {p.name}")
            except DecoderMissing as e:
                log(f"  [실패] 통화 {c.index}: {e}")
        for i, s in enumerate(tracker.orphan_streams, 1):
            if per_stream and s.packets and is_supported(s.main_codec):
                p = out / safe_name(f"orphan{i}_{s.src[0]}_{s.src[1]}_to_{s.dst[0]}_{s.dst[1]}", 80).replace(".", "-")
                p = p.with_name(p.name + ".wav")
                try:
                    save_stream_wav(s, p)
                    log(f"  {p.name}")
                except DecoderMissing as e:
                    log(f"  [실패] {e}")
    return out
