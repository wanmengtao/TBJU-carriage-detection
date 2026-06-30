#!/usr/bin/env python3
"""
LD3320 UART voice command mapping.

The LD3320 modules sold as "UART voice recognition" boards do not all use the
same payload format. Some send text, some send ASCII command IDs, and some send
binary bytes. This module keeps the matching rules configurable so the GUI can
be adjusted without code changes after the actual module is connected.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_COMMANDS = {
    "open_camera": {
        "label": "打开摄像头",
        "matches": ["打开摄像头", "camera", "hex:01"],
    },
    "start_detection": {
        "label": "开始/继续检测",
        "matches": ["开始检测", "继续检测", "启动检测", "开始", "继续", "start", "resume", "hex:04"],
    },
    "stop_detection": {
        "label": "停止检测",
        "matches": ["停止检测", "关闭检测", "停止", "stop", "hex:02"],
    },
    "pause_detection": {
        "label": "暂停检测",
        "matches": ["暂停检测", "暂停", "pause", "hex:03"],
    },
    "start_capacity": {
        "label": "开始评估",
        "matches": ["开始评估", "开始压测", "性能评估", "扩展评估", "capacity", "hex:05"],
    },
    "stop_capacity": {
        "label": "停止评估",
        "matches": ["停止评估", "停止压测", "hex:06"],
    },
    "toggle_recording": {
        "label": "开始/停止录制",
        "matches": ["开始录制", "停止录制", "录制", "record", "hex:07"],
    },
    "show_status": {
        "label": "系统状态",
        "matches": ["系统状态", "当前状态", "状态", "status", "hex:08"],
    },
    "mute_alarm": {
        "label": "静音报警",
        "matches": ["静音", "静音报警", "mute", "hex:09"],
    },
    "unmute_alarm": {
        "label": "解除静音",
        "matches": ["解除静音", "取消静音", "unmute", "hex:0a"],
    },
}


DEFAULT_CONFIG = {
    "serial": {
        "port_linux": "/dev/ttyS9",
        "port_windows": "COM3",
        "baudrate": 9600,
        "timeout_s": 0.2,
        "read_size": 64,
    },
    "protocol": {
        "ignore_syn6288_tts": True,
    },
    "commands": DEFAULT_COMMANDS,
}


@dataclass
class VoiceMatch:
    action: str
    label: str
    raw_text: str
    raw_hex: str
    matched_rule: str
    timestamp: str


@dataclass
class SerialFrameInfo:
    kind: str
    raw_hex: str
    text: str = ""
    command: Optional[int] = None
    encoding: Optional[int] = None
    length: Optional[int] = None


def ensure_default_config(path: Path) -> None:
    """Create a default config if it does not exist."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)


def load_voice_config(path: Optional[Path]) -> dict:
    if path is None:
        return DEFAULT_CONFIG.copy()
    path = Path(path)
    ensure_default_config(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        loaded = json.load(f)
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(loaded or {})
    if "commands" not in cfg or not cfg["commands"]:
        cfg["commands"] = DEFAULT_COMMANDS
    return cfg


def decode_payload(data: bytes) -> str:
    """Decode bytes as a readable string using common encodings."""
    if not data:
        return ""
    for enc in ("utf-8", "gbk", "gb2312", "ascii", "latin1"):
        try:
            text = data.decode(enc)
            return text.replace("\x00", "").strip()
        except UnicodeDecodeError:
            continue
    return ""


def decode_text_bytes(data: bytes, encoding_code: Optional[int] = None) -> str:
    """Decode text bytes from STC/SYN6288 style payloads."""
    encodings: List[str] = []
    if encoding_code == 0x00:
        encodings.extend(["gb2312", "gbk"])
    elif encoding_code == 0x01:
        encodings.extend(["gbk", "gb2312"])
    elif encoding_code == 0x02:
        encodings.append("big5")
    elif encoding_code == 0x03:
        encodings.append("utf-16-be")
    encodings.extend(["utf-8", "gbk", "gb2312", "latin1"])
    seen = set()
    for enc in encodings:
        if enc in seen:
            continue
        seen.add(enc)
        try:
            return data.decode(enc, errors="ignore").replace("\x00", "").strip()
        except LookupError:
            continue
    return ""


def compact_hex(data: bytes) -> str:
    return data.hex()


def command_hex(data: bytes) -> str:
    """Hex payload used for command matching, ignoring common line endings."""
    return data.strip(b"\r\n\x00").hex()


def parse_syn6288_tts_frame(data: bytes) -> Optional[SerialFrameInfo]:
    """Parse SYN6288-style TTS frames: FD len_hi len_lo cmd enc text...

    The user's LD3320+STC11 board can output these frames when the firmware is
    driving a speech playback module. They are feedback audio commands, not GUI
    control commands, so the matcher ignores them by default.
    """
    payload = data.strip(b"\r\n")
    if len(payload) < 5 or payload[0] != 0xFD:
        return None
    length = (payload[1] << 8) | payload[2]
    frame_body = payload[3:3 + length] if length else payload[3:]
    if len(frame_body) < 2:
        return SerialFrameInfo("syn6288_tts", compact_hex(data), length=length)
    command = frame_body[0]
    encoding = frame_body[1]
    text = decode_text_bytes(frame_body[2:], encoding)
    return SerialFrameInfo(
        kind="syn6288_tts",
        raw_hex=compact_hex(data),
        text=text,
        command=command,
        encoding=encoding,
        length=length,
    )


def describe_serial_frame(data: bytes) -> Optional[SerialFrameInfo]:
    return parse_syn6288_tts_frame(data)


def _norm_text(text: str) -> str:
    return "".join(text.lower().split())


def _norm_hex_rule(rule: str) -> str:
    s = rule.strip().lower()
    if s.startswith("hex:"):
        s = s[4:]
    return s.replace("0x", "").replace(" ", "").replace("-", "").replace(":", "")


class VoiceCommandMapper:
    """Map raw serial payloads to command actions."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or DEFAULT_CONFIG
        self.commands = self.config.get("commands") or DEFAULT_COMMANDS
        self.protocol = self.config.get("protocol") or {}

    @classmethod
    def from_file(cls, path: Optional[Path]) -> "VoiceCommandMapper":
        return cls(load_voice_config(path))

    def match(self, data: bytes) -> Optional[VoiceMatch]:
        frame = describe_serial_frame(data)
        if frame and frame.kind == "syn6288_tts":
            if self.protocol.get("ignore_syn6288_tts", True):
                return None
            raw_text = frame.text
        else:
            raw_text = decode_payload(data)

        raw_hex = compact_hex(data)
        cmd_hex = command_hex(data)
        text_norm = _norm_text(raw_text)
        hex_norm = cmd_hex.lower()

        rules = []
        for action, spec in self.commands.items():
            for rule in spec.get("matches", []):
                rule = str(rule).strip()
                if rule:
                    rules.append((action, spec.get("label", action), rule))

        # Pass 1: exact matches. This prevents a short rule like "静音" from
        # stealing a longer command such as "解除静音".
        for action, label, rule in rules:
            if rule.lower().startswith("hex:"):
                hx = _norm_hex_rule(rule)
                if hx and hex_norm == hx:
                    return VoiceMatch(action, label, raw_text, raw_hex, rule, _now())
            else:
                rule_text = _norm_text(rule)
                if rule_text and text_norm and rule_text == text_norm:
                    return VoiceMatch(action, label, raw_text, raw_hex, rule, _now())

        # Pass 2: relaxed text matches only. Hex commands must be explicit and
        # exact after CR/LF stripping to avoid matching serial line endings.
        for action, label, rule in rules:
            if rule.lower().startswith("hex:"):
                continue
            rule_text = _norm_text(rule)
            if not rule_text:
                continue
            # Short numeric rules such as "1" or "01" are common command IDs,
            # but LD3320/TTS prompt packets can contain text like "[v10]".
            # Keep numeric rules exact-only to avoid false triggers.
            if rule_text.isdigit():
                continue
            if rule_text and text_norm and rule_text in text_norm:
                return VoiceMatch(action, label, raw_text, raw_hex, rule, _now())
        return None

    def describe_commands(self) -> List[str]:
        lines = []
        for action, spec in self.commands.items():
            label = spec.get("label", action)
            rules = ", ".join(str(x) for x in spec.get("matches", []))
            lines.append(f"{action}: {label} [{rules}]")
        return lines


class LD3320SerialReader:
    """Small pyserial wrapper used by GUI and CLI test script."""

    def __init__(self, port: str, baudrate: int = 9600, timeout_s: float = 0.2,
                 read_size: int = 64):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self.read_size = int(read_size)
        self.serial = None

    def open(self):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("缺少 pyserial，请先执行: pip install pyserial") from exc
        self.serial = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout_s,
        )
        return self

    def close(self):
        if self.serial:
            self.serial.close()
            self.serial = None

    def read_payload(self) -> bytes:
        if not self.serial:
            return b""
        data = self.serial.readline()
        if data:
            return data
        return self.serial.read(self.read_size)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
