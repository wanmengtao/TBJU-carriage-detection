"""
板端命令轮询器 — 后台守护线程，轮询看板 /api/commands/pending 端点。
取到命令后通过回调执行，再 ack 回看板。仅依赖标准库。
"""

import threading
import time
import json
import urllib.request
import urllib.error
from datetime import datetime
from typing import Callable, Optional

from .utils import get_local_ipv4


class CommandPoller:
    def __init__(self, dashboard_url='http://192.168.1.100:8000',
                 device_id='ELF2-TBJU-01', poll_interval=2.0):
        self.dashboard_url = dashboard_url.rstrip('/')
        self.device_id = device_id
        self.poll_interval = poll_interval

        self._command_callback = None
        self._param_callback = None
        self._status_callback = None

        self._running = False
        self._thread = None
        self._heartbeat_thread = None

        self.commands_executed = 0
        self.last_error = None
        self.connected = False

    def on_command(self, callback: Callable):
        self._command_callback = callback

    def on_param_change(self, callback: Callable):
        self._param_callback = callback

    def on_status_change(self, callback: Callable):
        self._status_callback = callback

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, name='CommandPoller', daemon=True)
        self._thread.start()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name='Heartbeat', daemon=True)
        self._heartbeat_thread.start()
        print(f"[CommandPoller] 已启动，轮询间隔 {self.poll_interval}s，看板: {self.dashboard_url}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        print("[CommandPoller] 已停止")

    def close(self):
        self.stop()

    def _poll_loop(self):
        while self._running:
            try:
                url = f"{self.dashboard_url}/api/commands/pending?device_id={self.device_id}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    commands = json.loads(resp.read().decode()).get('commands', [])
                if not self.connected:
                    self.connected = True
                    self.last_error = None
                    if self._status_callback:
                        self._status_callback(True)
                for cmd in commands:
                    self._handle_command(cmd)
            except (urllib.error.URLError, OSError) as e:
                if self.connected:
                    self.connected = False
                    self.last_error = str(e)
                    if self._status_callback:
                        self._status_callback(False)
            except Exception as e:
                self.last_error = str(e)
            time.sleep(self.poll_interval)

    def _handle_command(self, cmd):
        action = cmd.get('action')
        params = cmd.get('params')
        command_id = cmd.get('command_id')
        if not action or not command_id:
            print(f"[CommandPoller] 命令缺少 action 或 command_id: {cmd}")
            return
        print(f"[远程控制] 收到命令: {action} (id={command_id[:8]}...)")
        try:
            if action == 'set_param' and params and self._param_callback:
                result = self._param_callback(params)
            elif self._command_callback:
                result = self._command_callback(action)
            else:
                result = {"ok": False, "message": "无回调注册"}

            ok = bool(result.get("ok", False)) if isinstance(result, dict) else bool(result)
            message = result.get("message", "") if isinstance(result, dict) else "executed"

            self._ack(command_id, ok, message)
            if ok:
                self.commands_executed += 1
        except Exception as e:
            self._ack(command_id, False, f'执行失败: {e}')

    def _ack(self, command_id, success, message):
        url = f"{self.dashboard_url}/api/commands/{command_id}/ack"
        payload = json.dumps({'success': success, 'message': message}).encode()
        req = urllib.request.Request(url, data=payload, method='POST')
        req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as e:
            print(f"[CommandPoller] ack 失败: {e}")

    def _heartbeat_loop(self):
        while self._running:
            try:
                self._send_heartbeat()
            except Exception:
                pass
            time.sleep(10)

    def _send_heartbeat(self):
        url = f"{self.dashboard_url}/api/commands/heartbeat"
        payload = json.dumps({
            'device_id': self.device_id,
            'ip': self._get_local_ip(),
            'status': 'running',
            'commands_executed': self.commands_executed,
        }).encode()
        req = urllib.request.Request(url, data=payload, method='POST')
        req.add_header('Content-Type', 'application/json')
        urllib.request.urlopen(req, timeout=5)

    @staticmethod
    def _get_local_ip():
        return get_local_ipv4()
