#!/usr/bin/env python3
"""Non-blocking event uploader with local offline queue.

The GUI can enqueue detection events immediately after inference. Events are
stored as JSON files first, then a background worker uploads them to the PC
dashboard when network telemetry is enabled. If WiFi or the server is down, the
pending files stay on disk and are retried later.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional
from urllib import error, request

from .utils import get_local_ipv4, validate_url

DEFAULT_DEVICE_ID = "ELF2-TBJU-01"
DEFAULT_SERVER_URL = os.environ.get(
    "TBJU_EVENT_SERVER_URL",
    "",  # 默认空，需在 GUI 或环境变量中配置
)

MAX_PAYLOAD_BYTES = 512 * 1024  # 512KB
MAX_SPOOL_BYTES = 200 * 1024 * 1024  # 200MB


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _post_json(url: str, payload: Dict, timeout_s: float = 2.0) -> Dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "TBJU-ELF2-Uploader/1.0",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        text = resp.read(4096).decode("utf-8", errors="ignore")
        return {"status": resp.status, "body": text}


class EventUploader:
    """Disk-backed uploader that never blocks the inference thread."""

    def __init__(
        self,
        spool_dir: Path,
        server_url: str = DEFAULT_SERVER_URL,
        enabled: bool = False,
        timeout_s: float = 2.0,
        retry_interval_s: float = 3.0,
        max_pending: int = 300,
    ):
        self.spool_dir = Path(spool_dir)
        self.pending_dir = self.spool_dir / "pending"
        self.sent_dir = self.spool_dir / "sent"
        self.failed_dir = self.spool_dir / "failed"
        for path in (self.pending_dir, self.sent_dir, self.failed_dir):
            path.mkdir(parents=True, exist_ok=True)

        # 清理上次崩溃留下的 .tmp 文件
        for tmp_file in self.pending_dir.glob("*.tmp"):
            tmp_file.unlink(missing_ok=True)

        self.server_url = server_url
        self.enabled = bool(enabled)
        self.timeout_s = float(timeout_s)
        self.retry_interval_s = float(retry_interval_s)
        self.max_pending = max(1, int(max_pending))

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._last_error = ""
        self._last_success = ""
        self._last_attempt = ""
        self._success_count = 0
        self._failure_count = 0
        self._dropped_count = 0

        # 带宽统计
        self._bytes_uploaded = 0
        self._bytes_uploaded_total = 0
        self._bandwidth_reset_ts = time.monotonic()

        self._thread = None

    def start(self) -> None:
        """显式启动上传线程。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="TBJUEventUploader", daemon=True)
        self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = bool(enabled)
        self._wake.set()

    def set_server_url(self, url: str) -> bool:
        """设置服务器 URL，返回是否生效。"""
        cleaned = re.sub(r'[\x00-\x1f\x7f]', '', (url or "")).strip()
        if cleaned and not validate_url(cleaned):
            return False
        with self._lock:
            self.server_url = cleaned
        self._wake.set()
        return True

    def enqueue(self, payload: Dict) -> Path:
        event = dict(payload)
        event.setdefault("event_id", f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}")
        event.setdefault("created_at", now_ts())
        event.setdefault("device_id", DEFAULT_DEVICE_ID)
        event.setdefault("board_ip", get_local_ipv4())

        # 磁盘容量限制
        self._trim_spool_size()

        # payload 大小限制：超限时丢弃 thumbnail
        body = json.dumps(event, ensure_ascii=False, indent=2)
        if len(body.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            event.pop("thumbnail", None)
            body = json.dumps(event, ensure_ascii=False, indent=2)

        self._trim_pending_queue()

        name = f"{int(time.time() * 1000)}_{event['event_id']}.json"
        path = self.pending_dir / _safe_name(name)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)
        self._trim_spool_size()  # 写入后再次检查，防止刚好超过限额
        self._wake.set()
        return path

    def test_connection(self) -> Dict:
        payload = {
            "event_id": f"test-{int(time.time())}",
            "device_id": DEFAULT_DEVICE_ID,
            "event_type": "network_test",
            "severity": "info",
            "created_at": now_ts(),
            "board_ip": get_local_ipv4(),
            "message": "ELF2 board network upload test",
        }
        with self._lock:
            url = self.server_url
        if not url:
            return {"ok": False, "error": "server URL is empty"}
        try:
            resp = _post_json(url, payload, timeout_s=self.timeout_s)
            return {"ok": 200 <= int(resp["status"]) < 300, "response": resp}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def flush_once(self) -> bool:
        """Upload one pending event if possible."""
        with self._lock:
            enabled = self.enabled
            url = self.server_url
        if not enabled or not url:
            return False

        files = sorted(self.pending_dir.glob("*.json"))
        if not files:
            return False

        path = files[0]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            body_bytes = path.stat().st_size
            with self._lock:
                self._last_attempt = now_ts()
            _post_json(url, payload, timeout_s=self.timeout_s)
            sent_path = self.sent_dir / path.name
            shutil.move(str(path), str(sent_path))
            self._cleanup_dir(self.sent_dir, keep=1000)
            self._cleanup_dir(self.failed_dir, keep=2000)
            with self._lock:
                self._success_count += 1
                self._last_success = now_ts()
                self._last_error = ""
                self._bytes_uploaded += body_bytes
                self._bytes_uploaded_total += body_bytes
            return True
        except Exception as exc:
            with self._lock:
                self._failure_count += 1
                self._last_error = str(exc)
            return False

    def get_bandwidth(self) -> Dict:
        """返回带宽统计（bytes/sec 和总量）。"""
        now = time.monotonic()
        with self._lock:
            elapsed = max(now - self._bandwidth_reset_ts, 1e-6)
            bps = self._bytes_uploaded / elapsed
            total = self._bytes_uploaded_total
            # 每 10 秒重置瞬时速率
            if elapsed > 10.0:
                self._bytes_uploaded = 0
                self._bandwidth_reset_ts = now
        return {
            "bytes_per_sec": round(bps, 1),
            "total_bytes": total,
            "total_kb": round(total / 1024, 1),
        }

    def upload_csv_file(self, file_path: Path, session_name: str = "", file_type: str = "result") -> bool:
        """将 CSV 文件上传到看板的 /api/files/upload 接口。"""
        with self._lock:
            url = self.server_url
        if not url:
            return False

        file_path = Path(file_path)
        if not file_path.exists():
            return False

        # 从 URL 推导出 /api/files/upload 地址
        if "/api/events" in url:
            base_url = url.rsplit("/api/events", 1)[0]
        else:
            base_url = url.rstrip("/")
        upload_url = f"{base_url}/api/files/upload"

        try:
            import urllib.request
            import urllib.parse

            boundary = "----TBJUFormBoundary" + uuid.uuid4().hex[:16]
            device_id = DEFAULT_DEVICE_ID
            filename = file_path.name

            with open(file_path, "rb") as f:
                file_data = f.read()

            body = b""
            for key, val in [("device_id", device_id), ("session_name", session_name), ("file_type", file_type)]:
                body += f"--{boundary}\r\n".encode()
                body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
                body += f"{val}\r\n".encode()

            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
            body += b"Content-Type: text/csv\r\n\r\n"
            body += file_data
            body += b"\r\n"
            body += f"--{boundary}--\r\n".encode()

            req = urllib.request.Request(
                upload_url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("ok"):
                    with self._lock:
                        self._bytes_uploaded += len(body)
                        self._bytes_uploaded_total += len(body)
                    return True
            return False
        except Exception as e:
            print(f"[EventUploader] CSV 上传失败: {e}")
            return False

    def status(self) -> Dict:
        with self._lock:
            data = {
                "enabled": self.enabled,
                "server_url": self.server_url,
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "dropped_count": self._dropped_count,
                "max_pending": self.max_pending,
                "last_success": self._last_success,
                "last_error": self._last_error,
                "last_attempt": self._last_attempt,
            }
        data["pending_count"] = len(list(self.pending_dir.glob("*.json")))
        data["sent_count"] = len(list(self.sent_dir.glob("*.json")))
        data["board_ip"] = get_local_ipv4()
        return data

    @staticmethod
    def _cleanup_dir(directory: Path, keep: int = 50):
        files = sorted(directory.glob("*.json"), key=lambda f: f.stat().st_mtime)
        if len(files) > keep:
            for f in files[:len(files) - keep]:
                f.unlink(missing_ok=True)

    def _trim_spool_size(self) -> None:
        """限制 spool 目录总大小，超出时删除最旧的文件。"""
        def _dir_size(path: Path) -> int:
            return sum(f.stat().st_size for f in path.rglob("*.json") if f.is_file())

        while _dir_size(self.spool_dir) > MAX_SPOOL_BYTES:
            candidates = sorted(
                list(self.pending_dir.glob("*.json")) +
                list(self.sent_dir.glob("*.json")) +
                list(self.failed_dir.glob("*.json")),
                key=lambda p: p.stat().st_mtime,
            )
            if not candidates:
                break
            candidates[0].unlink(missing_ok=True)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            uploaded = self.flush_once()
            if uploaded:
                continue
            self._wake.wait(self.retry_interval_s)
            self._wake.clear()

    def _trim_pending_queue(self) -> None:
        def _mtime(item: Path) -> float:
            try:
                return item.stat().st_mtime
            except OSError:
                return 0.0

        files = sorted(self.pending_dir.glob("*.json"), key=_mtime)
        overflow = len(files) - self.max_pending + 1
        if overflow <= 0:
            return

        dropped = 0
        for path in files[:overflow]:
            if not path.exists():
                continue
            target = self.failed_dir / f"dropped_{path.name}"
            if target.exists():
                target = self.failed_dir / f"dropped_{uuid.uuid4().hex[:8]}_{path.name}"
            try:
                shutil.move(str(path), str(target))
                dropped += 1
            except Exception:
                try:
                    path.unlink(missing_ok=True)
                    dropped += 1
                except Exception:
                    pass

        if dropped:
            with self._lock:
                self._dropped_count += dropped


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
