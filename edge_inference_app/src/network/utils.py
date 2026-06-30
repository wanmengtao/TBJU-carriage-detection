"""公共网络工具函数，供 event_uploader 和 command_poller 共用。"""

import socket


def get_local_ipv4() -> str:
    """获取本机 IPv4 地址（通过 UDP 连接推断出口 IP）。"""
    for target in ("223.5.5.5", "8.8.8.8", "192.168.1.1"):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.5)
            sock.connect((target, 80))
            ip = sock.getsockname()[0]
            sock.close()
            return ip
        except Exception:
            continue
    return "127.0.0.1"


def validate_url(url: str) -> bool:
    """校验 URL 格式，必须以 http:// 或 https:// 开头。"""
    if not url:
        return False
    url = url.strip().lower()
    return url.startswith("http://") or url.startswith("https://")
