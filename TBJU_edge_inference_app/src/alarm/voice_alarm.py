#!/usr/bin/env python3
"""
voice_alarm.py

Non-blocking voice alarm playback for the TBJU GUI.

The detector submits short alarm events from the GUI thread. A background
worker plays either a bundled audio file or falls back to the operating
system's text-to-speech command. Event-level cooldowns prevent continuous
frames from producing noisy repeated alerts.
"""

import queue
import shutil
import subprocess
import sys
import threading
import time
import os
import shlex
from pathlib import Path
from typing import Dict, Optional


class VoiceAlarmManager:
    """Small background audio player with per-event cooldown."""

    AUDIO_EXTS = ('.wav', '.mp3', '.ogg')

    DEFAULT_AUDIO_NAMES = {
        'test': 'alarm_test',
        'debris': 'alarm_debris',
        'track_debris': 'alarm_track_debris',
        'carriage_debris': 'alarm_carriage_debris',
        'temp_high': 'alarm_temp_high',
        'temp_critical': 'alarm_temp_critical',
        'cpu_high': 'alarm_cpu_high',
        'memory_high': 'alarm_memory_high',
    }

    def __init__(self, audio_dir: Optional[Path] = None, enabled: bool = True):
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.enabled = enabled
        self.alsa_device = os.environ.get('TBJU_ALARM_ALSA_DEVICE', '').strip()
        self._last_play: Dict[str, float] = {}
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=50)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name='VoiceAlarmPlayer', daemon=True)
        self._thread.start()

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def trigger(
        self,
        event_key: str,
        text: str,
        cooldown_s: float = 10.0,
        force: bool = False,
        repeat_count: int = 1,
        repeat_gap_s: float = 0.25,
    ) -> bool:
        """Queue an alarm if enabled and outside cooldown.

        Returns True only when the event was actually queued.
        """
        if not self.enabled and not force:
            return False

        now = time.monotonic()
        last = self._last_play.get(event_key, 0.0)
        if not force and now - last < cooldown_s:
            return False

        # 先更新冷却时间，再入队，防止并发触发
        self._last_play[event_key] = now

        try:
            repeat_count = max(1, int(repeat_count))
            repeat_gap_s = max(0.0, float(repeat_gap_s))
            self._queue.put_nowait((event_key, text, repeat_count, repeat_gap_s))
        except queue.Full:
            return False

        return True

    def close(self):
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            if len(item) >= 4:
                event_key, text, repeat_count, repeat_gap_s = item
            else:
                event_key, text = item
                repeat_count, repeat_gap_s = 1, 0.25
            try:
                for i in range(repeat_count):
                    if self._stop_event.is_set():
                        break
                    self._play(event_key, text)
                    if i < repeat_count - 1 and repeat_gap_s > 0:
                        time.sleep(repeat_gap_s)
            finally:
                self._queue.task_done()

    def _play(self, event_key: str, text: str):
        audio_file = self._find_audio_file(event_key)
        if audio_file and self._play_file(audio_file):
            return
        self._speak_text(text)

    def _find_audio_file(self, event_key: str) -> Optional[Path]:
        if not self.audio_dir or not self.audio_dir.exists():
            return None

        base = self.DEFAULT_AUDIO_NAMES.get(event_key, event_key)
        for ext in self.AUDIO_EXTS:
            p = self.audio_dir / f'{base}{ext}'
            if p.exists():
                return p
        return None

    def _play_file(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        try:
            if sys.platform == 'win32' and suffix == '.wav':
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME)
                return True

            players = []
            quoted_path = shlex.quote(str(path))
            players.append(('gst-play-1.0', f'gst-play-1.0 {quoted_path}', True))
            if suffix == '.wav':
                if self.alsa_device:
                    players.append(('aplay', ['aplay', '-q', '-D', self.alsa_device, str(path)]))
                players.extend([
                    ('paplay', ['paplay', str(path)]),
                    ('aplay', ['aplay', '-q', str(path)]),
                ])
            if self.alsa_device:
                players.append((
                    'gst-play-1.0',
                    [
                        'gst-play-1.0',
                        str(path),
                        f'--audiosink=alsasink device={self.alsa_device}',
                    ],
                    False,
                ))
            players.append(('gst-play-1.0', ['gst-play-1.0', str(path)], False))
            players.extend([
                ('ffplay', ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', str(path)], False),
                ('mpv', ['mpv', '--really-quiet', '--no-video', str(path)], False),
            ])

            for item in players:
                name, cmd = item[0], item[1]
                use_shell = bool(item[2]) if len(item) > 2 else False
                if shutil.which(name):
                    proc = subprocess.run(
                        cmd,
                        shell=use_shell,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=12,
                    )
                    if proc.returncode == 0:
                        return True
        except Exception:
            return False
        return False

    def _speak_text(self, text: str) -> bool:
        if not text:
            return False

        if sys.platform == 'win32':
            return self._speak_windows(text)

        if sys.platform == 'darwin':
            return self._run_tts(['say', text])

        linux_candidates = [
            ['spd-say', text],
            ['espeak-ng', '-v', 'zh', text],
            ['espeak', '-v', 'zh', text],
        ]
        for cmd in linux_candidates:
            if shutil.which(cmd[0]) and self._run_tts(cmd):
                return True
        return False

    def _speak_windows(self, text: str) -> bool:
        powershell = shutil.which('powershell') or shutil.which('powershell.exe')
        if not powershell:
            return False
        script = (
            'Add-Type -AssemblyName System.Speech; '
            '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            '$s.Volume = 100; $s.Rate = 0; '
            '$t = [Console]::In.ReadToEnd(); '
            '$s.Speak($t);'
        )
        try:
            subprocess.run(
                [powershell, '-NoProfile', '-Command', script],
                input=text,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _run_tts(cmd) -> bool:
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            return True
        except Exception:
            return False
