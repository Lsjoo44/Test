"""SIP 캡처 GUI (Windows 11 + Npcap).

  python sipcap_gui.py            # 또는 빌드한 SIPCapture.exe 더블클릭
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from sipcap import __version__
from sipcap.calls import Call, CallTracker
from sipcap.codecs import g729_decoder_status, get_g729_decoder, is_supported
from sipcap.export import (DecoderMissing, call_filename, export_all, safe_name, save_call_wav,
                           save_stream_wav, write_sip_log)

IS_WIN = os.name == "nt"
DEFAULT_DIR = Path.home() / "SIPCapture"

CALL_COLS = [("no", "No", 40), ("start", "시작 시각", 140), ("from", "발신", 150), ("to", "수신", 150),
             ("state", "상태", 90), ("dur", "통화시간", 70), ("codec", "코덱", 90), ("pkts", "RTP 패킷", 70),
             ("lost", "손실", 50), ("callid", "Call-ID", 220)]
STREAM_COLS = [("dir", "방향", 110), ("src", "출발", 160), ("dst", "도착", 160), ("ssrc", "SSRC", 90),
               ("codec", "코덱", 80), ("pkts", "패킷", 60), ("lost", "손실", 50), ("loss", "손실률", 60),
               ("jitter", "지터(ms)", 70), ("dur", "길이(s)", 60)]
EVENT_COLS = [("time", "시각", 100), ("src", "출발", 160), ("dst", "도착", 160), ("proto", "", 40),
              ("line", "메시지", 420)]

DIR_LABEL = {"caller->callee": "발신 → 수신", "callee->caller": "수신 → 발신", "?": "?"}


def npcap_installed() -> bool:
    if not IS_WIN:
        return True
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return os.path.isdir(os.path.join(root, "System32", "Npcap"))


def fmt_dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:d}:{sec % 60:02d}"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"SIP Capture v{__version__}  —  G.711 / G.729")
        self.geometry("1180x760")
        self.minsize(900, 560)
        self.tracker = CallTracker()
        self.capture = None
        self.ifaces: list[tuple[str, str]] = []
        self._shown_version = -1
        self._loading = False
        self._play_file: str | None = None
        self._load_result = None

        self.var_iface = tk.StringVar()
        self.var_filter = tk.StringVar()
        self.var_heur = tk.BooleanVar(value=False)
        self.var_savepcap = tk.BooleanVar(value=True)
        self.var_status = tk.StringVar(value="준비")
        self.var_g729 = tk.StringVar(value=g729_decoder_status())

        self._build()
        self.after(100, self._startup_checks)
        self.after(500, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build(self):
        from sipcap.capture import DEFAULT_BPF
        self.var_filter.set(DEFAULT_BPF)
        pad = dict(padx=4, pady=3)

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(top, text="네트워크 장치").grid(row=0, column=0, sticky="w", **pad)
        self.cb_iface = ttk.Combobox(top, textvariable=self.var_iface, state="readonly", width=80)
        self.cb_iface.grid(row=0, column=1, columnspan=4, sticky="we", **pad)
        ttk.Button(top, text="새로고침", command=self._load_ifaces).grid(row=0, column=5, **pad)
        ttk.Label(top, text="캡처 필터(BPF)").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.var_filter, width=40).grid(row=1, column=1, sticky="we", **pad)
        ttk.Checkbutton(top, text="SIP 없이 RTP 추측", variable=self.var_heur).grid(row=1, column=2, sticky="w", **pad)
        ttk.Checkbutton(top, text="pcap 자동 저장", variable=self.var_savepcap).grid(row=1, column=3, sticky="w", **pad)
        top.columnconfigure(1, weight=1)

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        self.btn_start = ttk.Button(bar, text="▶ 캡처 시작", command=self.start_capture)
        self.btn_stop = ttk.Button(bar, text="■ 중지", command=self.stop_capture, state="disabled")
        self.btn_open = ttk.Button(bar, text="pcap 열기...", command=self.open_pcap)
        for b in (self.btn_start, self.btn_stop, self.btn_open):
            b.pack(side="left", padx=3)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(bar, text="▶ 재생", command=self.play_selected).pack(side="left", padx=3)
        ttk.Button(bar, text="■ 재생 정지", command=self.stop_play).pack(side="left", padx=3)
        ttk.Button(bar, text="통화 WAV 저장...", command=self.save_selected_wav).pack(side="left", padx=3)
        ttk.Button(bar, text="전체 내보내기...", command=self.export_everything).pack(side="left", padx=3)
        ttk.Button(bar, text="목록 지우기", command=self.clear).pack(side="right", padx=3)
        ttk.Button(bar, text="결과 폴더", command=lambda: self._open_folder(DEFAULT_DIR)).pack(side="right", padx=3)

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=6, pady=2)

        f_calls = ttk.LabelFrame(paned, text="통화 목록")
        self.tv_calls = self._make_tree(f_calls, CALL_COLS, height=10)
        self.tv_calls.bind("<<TreeviewSelect>>", lambda e: self._refresh_detail(force=True))
        self.tv_calls.bind("<Double-1>", lambda e: self.play_selected())
        self.tv_calls.bind("<Button-3>", self._call_menu)
        paned.add(f_calls, weight=3)

        nb = self.nb = ttk.Notebook(paned)
        f_streams = ttk.Frame(nb)
        self.tv_streams = self._make_tree(f_streams, STREAM_COLS, height=5)
        self.tv_streams.bind("<Button-3>", self._stream_menu)
        nb.add(f_streams, text="RTP 스트림")
        f_sip = ttk.Frame(nb)
        sp = ttk.PanedWindow(f_sip, orient="horizontal")
        sp.pack(fill="both", expand=True)
        f_ev = ttk.Frame(sp)
        self.tv_events = self._make_tree(f_ev, EVENT_COLS, height=6)
        self.tv_events.bind("<<TreeviewSelect>>", self._show_event)
        sp.add(f_ev, weight=3)
        f_raw = ttk.Frame(sp)
        self.txt_raw = tk.Text(f_raw, wrap="none", height=8, font=("Consolas", 9))
        ys = ttk.Scrollbar(f_raw, orient="vertical", command=self.txt_raw.yview)
        self.txt_raw.configure(yscrollcommand=ys.set)
        self.txt_raw.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        sp.add(f_raw, weight=2)
        nb.add(f_sip, text="SIP 메시지")
        paned.add(nb, weight=2)

        st = ttk.Frame(self, relief="sunken")
        st.pack(fill="x", side="bottom")
        ttk.Label(st, textvariable=self.var_status, anchor="w").pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(st, textvariable=self.var_g729, anchor="e").pack(side="right", padx=6)

        self.menu_call = tk.Menu(self, tearoff=0)
        self.menu_call.add_command(label="재생", command=self.play_selected)
        self.menu_call.add_command(label="통화 WAV 저장 (스테레오: 왼쪽 발신 / 오른쪽 수신)...", command=self.save_selected_wav)
        self.menu_call.add_command(label="SIP 메시지 원문 저장...", command=self.save_selected_sip)
        self.menu_stream = tk.Menu(self, tearoff=0)
        self.menu_stream.add_command(label="이 스트림만 WAV 저장 (mono)...", command=self.save_selected_stream)

    def _make_tree(self, parent, cols, height):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tv = ttk.Treeview(frame, columns=[c[0] for c in cols], show="headings", height=height)
        for key, label, width in cols:
            tv.heading(key, text=label)
            tv.column(key, width=width, minwidth=30, stretch=key in ("from", "to", "callid", "line", "src", "dst"),
                      anchor="w" if key in ("from", "to", "callid", "line", "src", "dst", "start", "state", "dir") else "center")
        ys = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        tv.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tv.tag_configure("active", foreground="#0a7d2c")
        tv.tag_configure("failed", foreground="#b3261e")
        return tv

    # ------------------------------------------------------------------ 시작 확인
    def _startup_checks(self):
        if not npcap_installed():
            if messagebox.askyesno("Npcap 필요",
                                   "실시간 캡처에는 Npcap 이 필요합니다 (Windows 11 은 WinPcap 미지원).\n"
                                   "설치 시 'WinPcap API-compatible Mode' 를 체크하세요.\n\n"
                                   "다운로드 페이지를 열까요?\n(저장된 pcap 파일 분석은 Npcap 없이도 됩니다)"):
                webbrowser.open("https://npcap.com/#download")
        self._load_ifaces()
        if get_g729_decoder() is None:
            self.var_g729.set("⚠ G.729 디코더 없음 → ffmpeg.exe 를 프로그램 폴더에 두세요 (G.711 은 동작)")

    def _load_ifaces(self):
        try:
            from sipcap.capture import default_interface, list_interfaces
            self.ifaces = list_interfaces()
            d = default_interface()
        except Exception as e:                       # scapy/Npcap 문제
            self.ifaces, d = [], None
            self.var_status.set(f"장치 목록을 못 읽음: {e}")
        self.cb_iface["values"] = [lab for _, lab in self.ifaces]
        idx = next((i for i, (n, _) in enumerate(self.ifaces) if n == d), 0)
        if self.ifaces:
            self.cb_iface.current(idx)

    # ------------------------------------------------------------------ 캡처
    def start_capture(self):
        from sipcap.capture import LiveCapture
        if not self.ifaces:
            messagebox.showerror("장치 없음", "캡처할 네트워크 장치가 없습니다. Npcap 설치 후 '새로고침' 하세요.")
            return
        iface = self.ifaces[self.cb_iface.current()][0]
        pcap = None
        if self.var_savepcap.get():
            DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
            pcap = str(DEFAULT_DIR / f"capture_{datetime.now():%Y%m%d_%H%M%S}.pcap")
        self.tracker.rtp_heuristic = self.var_heur.get()
        tr = self.tracker
        cap = LiveCapture(iface, lambda tup: tr.process_packet(*tup), self.var_filter.get().strip(), pcap)
        try:
            cap.start()
        except Exception as e:
            messagebox.showerror("캡처 시작 실패",
                                 f"{e}\n\n확인: Npcap 설치 / 장치 선택 / 필터 문법 / (필요 시) 관리자 권한으로 실행")
            return
        self.capture = cap
        self._pcap_path = pcap
        if cap.warning:
            messagebox.showwarning("캡처 필터", cap.warning)
        self.btn_start.configure(state="disabled")
        self.btn_open.configure(state="disabled")
        self.btn_stop.configure(state="normal")

    def stop_capture(self):
        if self.capture:
            self.capture.stop()
            n = self.capture.captured
            self.capture = None
            msg = f"캡처 중지 — {n} 패킷"
            if self._pcap_path:
                msg += f"  (pcap: {self._pcap_path})"
            self.var_status.set(msg)
        self.btn_start.configure(state="normal")
        self.btn_open.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def open_pcap(self):
        path = filedialog.askopenfilename(title="pcap 파일 열기",
                                          filetypes=[("Capture", "*.pcap *.pcapng *.cap"), ("All", "*.*")])
        if not path:
            return
        from sipcap.capture import read_pcap
        self.clear()
        self.tracker.rtp_heuristic = self.var_heur.get()
        self._loading = True
        self.btn_start.configure(state="disabled")
        tr = self.tracker

        def work():                    # Tk 는 다른 스레드에서 호출하면 안 됨 → 결과만 남기고 _tick 이 처리
            err = None
            try:
                n = read_pcap(path, tr)
            except Exception as e:
                err, n = e, 0
            self._load_result = (path, n, err)

        threading.Thread(target=work, daemon=True).start()
        self.var_status.set(f"읽는 중: {path}")

    def _loaded(self, path, n, err):
        self._loading = False
        self.btn_start.configure(state="normal")
        if err:
            messagebox.showerror("pcap 읽기 실패", str(err))
        self._refresh(force=True)
        self.var_status.set(f"{Path(path).name}: {n} 패킷, 통화 {len(self.tracker.calls)}건")

    def clear(self):
        if self.capture:
            return
        self.tracker = CallTracker(rtp_heuristic=self.var_heur.get())
        self._shown_version = -1
        for tv in (self.tv_calls, self.tv_streams, self.tv_events):
            tv.delete(*tv.get_children())
        self.txt_raw.delete("1.0", "end")

    # ------------------------------------------------------------------ 화면 갱신
    def _tick(self):
        try:
            if self._load_result is not None:
                res, self._load_result = self._load_result, None
                self._loaded(*res)
            self._refresh()
        finally:
            self.after(700, self._tick)

    def _refresh(self, force=False):
        tr = self.tracker
        if self.capture:
            if not self.capture.running:                  # 캡처 스레드가 죽음
                err = getattr(self.capture._sniffer, "exception", None)
                self.stop_capture()
                if err:
                    messagebox.showerror("캡처 중단", str(err))
            else:
                self.var_status.set(f"● 캡처 중 — 패킷 {self.capture.captured}  |  SIP {tr.sip_count}  |  "
                                    f"RTP {tr.rtp_count}  |  통화 {len(tr.calls)}")
        elif self._loading:
            self.var_status.set(f"읽는 중... 패킷 {tr.packet_count}")
        if not force and tr.version == self._shown_version:
            return
        self._shown_version = tr.version
        with tr.lock:
            existing = set(self.tv_calls.get_children())
            for c in tr.calls.values():
                iid = str(c.index)
                vals = (c.index, c.start_str, c.from_, c.to, c.state, fmt_dur(c.duration) if c.connect_time else "",
                        c.codec, sum(len(s.packets) for s in c.streams), sum(s.lost for s in c.streams), c.call_id)
                tag = ("active",) if c.state == "연결됨" else (("failed",) if c.connect_time is None and c.end_time else ())
                if iid in existing:
                    self.tv_calls.item(iid, values=vals, tags=tag)
                else:
                    self.tv_calls.insert("", "end", iid=iid, values=vals, tags=tag)
        self._refresh_detail()

    def _selected_call(self) -> Call | None:
        sel = self.tv_calls.selection()
        if not sel:
            return None
        idx = int(sel[0])
        with self.tracker.lock:
            for c in self.tracker.calls.values():
                if c.index == idx:
                    return c
        return None

    def _refresh_detail(self, force=False):
        c = self._selected_call()
        if c is None:
            return
        with self.tracker.lock:
            rows = []
            for i, s in enumerate(c.streams):
                exp = s.expected
                rows.append((str(i), (DIR_LABEL.get(s.direction, s.direction), f"{s.src[0]}:{s.src[1]}",
                             f"{s.dst[0]}:{s.dst[1]}", f"0x{s.ssrc:08X}", s.main_codec, len(s.packets), s.lost,
                             f"{100.0 * s.lost / exp:.1f}%" if exp else "", f"{s.jitter_ms:.2f}",
                             f"{s.last_time - s.first_time:.1f}")))
            events = [(str(i), (datetime.fromtimestamp(e.time).strftime("%H:%M:%S.%f")[:-3], e.src, e.dst,
                                e.transport, e.start_line)) for i, e in enumerate(c.events)]
        self._sync_tree(self.tv_streams, rows, force)
        self._sync_tree(self.tv_events, events, force)

    @staticmethod
    def _sync_tree(tv, rows, force):
        if force:
            tv.delete(*tv.get_children())
        existing = set(tv.get_children())
        for iid, vals in rows:
            if iid in existing:
                tv.item(iid, values=vals)
            else:
                tv.insert("", "end", iid=iid, values=vals)

    def _show_event(self, _e=None):
        c = self._selected_call()
        sel = self.tv_events.selection()
        if c is None or not sel:
            return
        ev = c.events[int(sel[0])]
        self.txt_raw.delete("1.0", "end")
        self.txt_raw.insert("1.0", ev.raw.decode("utf-8", "replace").replace("\r\n", "\n"))

    # ------------------------------------------------------------------ 저장 / 재생
    def _call_menu(self, e):
        iid = self.tv_calls.identify_row(e.y)
        if iid:
            self.tv_calls.selection_set(iid)
            self.menu_call.tk_popup(e.x_root, e.y_root)

    def _stream_menu(self, e):
        iid = self.tv_streams.identify_row(e.y)
        if iid:
            self.tv_streams.selection_set(iid)
            self.menu_stream.tk_popup(e.x_root, e.y_root)

    def _need_call(self) -> Call | None:
        c = self._selected_call()
        if c is None:
            messagebox.showinfo("통화 선택", "먼저 통화 목록에서 통화를 선택하세요.")
            return None
        if not any(s.packets for s in c.streams):
            messagebox.showinfo("음성 없음", "이 통화에는 캡처된 RTP 음성이 없습니다.")
            return None
        if not any(is_supported(s.main_codec) for s in c.streams):
            messagebox.showwarning("코덱 미지원", f"지원하지 않는 코덱입니다: {c.codec}\n(지원: G.711 PCMU/PCMA, G.729)")
            return None
        return c

    def _run_export(self, fn, *args) -> bool:
        try:
            with self.tracker.lock:
                fn(*args)
            return True
        except DecoderMissing as e:
            messagebox.showerror("G.729 디코더 없음", str(e))
        except Exception as e:
            messagebox.showerror("저장 실패", str(e))
        return False

    def save_selected_wav(self):
        c = self._need_call()
        if c is None:
            return
        DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(title="통화 WAV 저장", initialdir=DEFAULT_DIR,
                                            initialfile=call_filename(c), defaultextension=".wav",
                                            filetypes=[("WAV", "*.wav")])
        if path and self._run_export(save_call_wav, c, path):
            self.var_status.set(f"저장: {path}")

    def save_selected_stream(self):
        c = self._selected_call()
        sel = self.tv_streams.selection()
        if c is None or not sel:
            return
        s = c.streams[int(sel[0])]
        name = call_filename(c).replace(".wav", f"_{safe_name(s.direction.replace('->', '_to_'))}.wav")
        path = filedialog.asksaveasfilename(title="스트림 WAV 저장", initialdir=DEFAULT_DIR, initialfile=name,
                                            defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if path and self._run_export(save_stream_wav, s, path):
            self.var_status.set(f"저장: {path}")

    def save_selected_sip(self):
        c = self._selected_call()
        if c is None:
            return
        path = filedialog.asksaveasfilename(title="SIP 메시지 저장", initialdir=DEFAULT_DIR,
                                            initialfile=call_filename(c).replace(".wav", ".sip.txt"),
                                            defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path and self._run_export(write_sip_log, c, path):
            self.var_status.set(f"저장: {path}")

    def export_everything(self):
        if not self.tracker.calls:
            messagebox.showinfo("내보내기", "통화가 없습니다.")
            return
        DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
        d = filedialog.askdirectory(title="내보낼 폴더", initialdir=DEFAULT_DIR)
        if not d:
            return
        out = Path(d) / f"export_{datetime.now():%Y%m%d_%H%M%S}"
        lines = []
        try:
            export_all(self.tracker, out, per_stream=True, log=lines.append)
        except Exception as e:
            messagebox.showerror("내보내기 실패", str(e))
            return
        fails = [ln for ln in lines if "[실패]" in ln or "[건너뜀]" in ln]
        messagebox.showinfo("내보내기 완료", f"{out}\n\n파일 {len(lines) - 1}개" + ("\n\n" + "\n".join(fails) if fails else ""))
        self._open_folder(out)

    def play_selected(self):
        c = self._need_call()
        if c is None:
            return
        if not IS_WIN:
            messagebox.showinfo("재생", "재생은 Windows 에서만 지원합니다. WAV 로 저장해서 들어 보세요.")
            return
        import winsound
        self.stop_play()
        fd, path = tempfile.mkstemp(prefix="sipcap_", suffix=".wav")
        os.close(fd)
        if self._run_export(save_call_wav, c, path):
            self._play_file = path
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.var_status.set(f"재생 중: 통화 {c.index} ({c.from_} → {c.to})")

    def stop_play(self):
        if IS_WIN:
            import winsound
            winsound.PlaySound(None, 0)
        if self._play_file:
            try:
                os.remove(self._play_file)
            except OSError:
                pass
            self._play_file = None

    @staticmethod
    def _open_folder(path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if IS_WIN:
            os.startfile(str(path))          # noqa: S606  (탐색기로 폴더 열기)
        else:
            webbrowser.open(path.as_uri())

    def _on_close(self):
        if self.capture:
            self.capture.stop()
        self.stop_play()
        self.destroy()


def main():
    for name in ("stdout", "stderr"):            # PyInstaller --windowed exe 에서는 None → 로그 출력 시 오류 방지
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    if IS_WIN:
        try:                                     # 고해상도(125%/150%) 화면에서 글자 흐림 방지
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    app = App()
    style = ttk.Style(app)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app.mainloop()


if __name__ == "__main__":
    sys.exit(main())
