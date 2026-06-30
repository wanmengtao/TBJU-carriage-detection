"""
TBJU 轨道异物智能检测远程看板 - 主应用
========================================
启动命令: python app.py
访问地址: http://localhost:8000
板端上传: http://电脑端IP:8000/api/events
"""
import os
import json
import base64
import csv
import io
import socket
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import database as db

# 配置
HOST = "0.0.0.0"
PORT = 8000
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads", "thumbnails")
SYNCED_DIR = os.path.join(os.path.dirname(__file__), "uploads", "synced_files")
# 自动保存到本地 output 目录（与开发板目录结构对齐）
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
device_status = {}  # 设备在线状态（内存存储）

# 带宽统计（内存存储，滑动窗口 10 秒）
import threading
import time as _time

_bw_lock = threading.Lock()
_bw_window = []  # [(timestamp, bytes), ...]
_bw_total_bytes = 0
_bw_peak_bps = 0.0


def get_lan_ipv4s():
    """返回本机可供板端访问的候选 IPv4 地址。"""
    ips = set()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and not ip.startswith("169.254."):
                ips.add(ip)
    except Exception:
        pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.connect(("223.5.5.5", 80))
        ip = sock.getsockname()[0]
        if ip and not ip.startswith("127.") and not ip.startswith("169.254."):
            ips.add(ip)
    except Exception:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass

    return sorted(ips)

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SYNCED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建 FastAPI 应用
app = FastAPI(
    title="TBJU 远程看板",
    description="TBJU 列车车厢与轨道异物智能检测系统 - 远程事件接收与看板",
    version="1.0.0"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 带宽统计中间件
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse


class BandwidthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        # 统计请求体大小（文件上传用 Content-Length，避免读两次）
        content_length = request.headers.get("content-length")
        if "/api/files/upload" in str(request.url):
            req_bytes = int(content_length) if content_length else 0
        else:
            body = await request.body()
            req_bytes = len(body)
        now = _time.time()
        with _bw_lock:
            global _bw_total_bytes, _bw_peak_bps
            _bw_total_bytes += req_bytes
            _bw_window.append((now, req_bytes))
            # 清理 10 秒前的记录
            cutoff = now - 10.0
            _bw_window[:] = [(t, b) for t, b in _bw_window if t >= cutoff]
            # 计算瞬时速率
            if len(_bw_window) >= 2:
                span = _bw_window[-1][0] - _bw_window[0][0]
                if span > 0:
                    total_in_window = sum(b for _, b in _bw_window)
                    bps = total_in_window / span
                    if bps > _bw_peak_bps:
                        _bw_peak_bps = bps
        response = await call_next(request)
        return response


app.add_middleware(BandwidthMiddleware)

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 挂载上传目录（用于访问缩略图）
app.mount("/uploads", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "uploads")), name="uploads")


@app.on_event("startup")
async def startup():
    """应用启动时初始化数据库"""
    db.init_db()
    lan_ips = get_lan_ipv4s()
    print(f"[TBJU] 远程看板服务启动")
    print(f"[TBJU] 监听地址: http://{HOST}:{PORT}")
    print(f"[TBJU] 看板页面: http://localhost:{PORT}")
    if lan_ips:
        print("[TBJU] 局域网访问地址:")
        print("[TBJU] 如果出现多个地址，请选择与 ELF2 板端 IP 同网段的那个。")
        for ip in lan_ips:
            print(f"  看板页面: http://{ip}:{PORT}")
            print(f"  板端上传: http://{ip}:{PORT}/api/events")
    else:
        print(f"[TBJU] 未检测到局域网 IPv4，请用 ipconfig 查看电脑端 IP 后填写 http://电脑端IP:{PORT}/api/events")


@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 - 返回看板页面"""
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/events")
async def receive_event(request: Request):
    """
    接收 ELF2 板端上传的事件

    请求体: JSON 格式的事件数据
    返回: {"ok": true, "event_id": "xxx", "received_at": "2026-06-14 12:34:56"}
    """
    try:
        event_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "invalid payload"})

    if not isinstance(event_data, dict):
        raise HTTPException(status_code=400, detail={"ok": False, "error": "invalid payload"})

    # 处理缩略图
    thumbnail_path = None
    thumbnail_data = event_data.get("thumbnail")
    if thumbnail_data and isinstance(thumbnail_data, dict):
        try:
            # 解码 base64 图片
            img_base64 = thumbnail_data.get("data", "")
            if img_base64:
                # 生成文件名
                event_id = event_data.get("event_id", f"evt-{int(datetime.now().timestamp()*1000)}")
                fmt = thumbnail_data.get("format", "jpg")
                today = datetime.now().strftime("%Y%m%d")
                day_dir = os.path.join(UPLOAD_DIR, today)
                os.makedirs(day_dir, exist_ok=True)

                filename = f"{event_id}.{fmt}"
                filepath = os.path.join(day_dir, filename)

                # 保存图片
                img_bytes = base64.b64decode(img_base64)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                # 保存相对路径
                thumbnail_path = f"uploads/thumbnails/{today}/{filename}"
                print(f"[TBJU] 缩略图已保存: {thumbnail_path}")
        except Exception as e:
            print(f"[TBJU] 缩略图保存失败: {e}")

    # 存入数据库
    event_id = db.insert_event(event_data, thumbnail_path)

    received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[TBJU] 收到事件: {event_id} | 类型: {event_data.get('event_type')} | 设备: {event_data.get('device_id')}")

    return JSONResponse(content={
        "ok": True,
        "event_id": event_id,
        "received_at": received_at
    })


@app.get("/api/events/recent")
async def get_recent_events(
    limit: int = Query(default=50, ge=1, le=500),
    event_type: Optional[str] = Query(default=None)
):
    """
    获取最近事件列表

    参数:
        limit: 返回数量（默认50，最大500）
        event_type: 事件类型筛选（可选）
    """
    events = db.get_recent_events(limit=limit, event_type=event_type)

    # 解析 JSON 字段
    for event in events:
        for field in ["ocr_texts", "class_counts", "detections", "timing", "system"]:
            if event.get(field):
                try:
                    event[field] = json.loads(event[field])
                except Exception:
                    pass

    return JSONResponse(content={
        "ok": True,
        "events": events
    })


@app.get("/api/events/{event_id}")
async def get_event(event_id: str):
    """
    获取单个事件详情

    路径参数:
        event_id: 事件 ID
    """
    event = db.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail={"ok": False, "error": "event not found"})

    # 解析 JSON 字段
    for field in ["ocr_texts", "class_counts", "detections", "timing", "system"]:
        if event.get(field):
            try:
                event[field] = json.loads(event[field])
            except Exception:
                pass

    return JSONResponse(content={
        "ok": True,
        "event": event
    })


@app.get("/api/stats")
async def get_stats():
    """获取统计数据"""
    return JSONResponse(content=db.get_stats())


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    lan_ips = get_lan_ipv4s()
    return JSONResponse(content={
        "ok": True,
        "service": "TBJU remote dashboard",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host": HOST,
        "port": PORT,
        "lan_ips": lan_ips,
        "dashboard_urls": [f"http://{ip}:{PORT}" for ip in lan_ips],
        "upload_urls": [f"http://{ip}:{PORT}/api/events" for ip in lan_ips],
    })


@app.get("/api/bandwidth")
async def get_bandwidth():
    """返回接收带宽统计（KB/s、峰值、累计）。"""
    with _bw_lock:
        now = _time.time()
        cutoff = now - 10.0
        window = [(t, b) for t, b in _bw_window if t >= cutoff]
        if len(window) >= 2:
            span = window[-1][0] - window[0][0]
            total_in_window = sum(b for _, b in window)
            current_bps = total_in_window / span if span > 0 else 0
        else:
            current_bps = 0
        return JSONResponse(content={
            "ok": True,
            "bytes_per_sec": round(current_bps, 1),
            "kb_per_sec": round(current_bps / 1024, 1),
            "peak_bytes_per_sec": round(_bw_peak_bps, 1),
            "peak_kb_per_sec": round(_bw_peak_bps / 1024, 1),
            "total_bytes": _bw_total_bytes,
            "total_kb": round(_bw_total_bytes / 1024, 1),
            "total_mb": round(_bw_total_bytes / 1024 / 1024, 2),
        })


@app.post("/api/files/upload")
async def upload_file(request: Request):
    """接收开发板上传的 CSV 文件"""
    try:
        form = await request.form()
        file = form.get("file")
        device_id = form.get("device_id", "ELF2-TBJU-01")
        session_name = form.get("session_name", "unknown")
        file_type = form.get("file_type", "result")  # result / metrics / capacity

        if not file:
            return JSONResponse(status_code=400, content={"ok": False, "error": "no file"})

        # 按设备和日期组织目录
        today = datetime.now().strftime("%Y%m%d")
        dest_dir = os.path.join(SYNCED_DIR, device_id, today)
        os.makedirs(dest_dir, exist_ok=True)

        # 文件名：session名_类型_原始文件名（清理特殊字符）
        orig_name = getattr(file, 'filename', 'data.csv')
        # 只保留字母数字、点、下划线、横杠
        import re as _re
        safe_session = _re.sub(r'[^\w\-]', '_', str(session_name))
        safe_orig = _re.sub(r'[^\w\.\-]', '_', str(orig_name))
        safe_name = f"{safe_session}_{file_type}_{safe_orig}"
        dest_path = os.path.join(dest_dir, safe_name)

        content = await file.read()

        # 磁盘空间检查（至少保留 100MB）
        import shutil
        disk_free = shutil.disk_usage(dest_dir).free
        if disk_free < 100 * 1024 * 1024:
            return JSONResponse(status_code=507, content={"ok": False, "error": "磁盘空间不足"})

        with open(dest_path, "wb") as f:
            f.write(content)

        # 同时保存到 output 目录（按 session 名归档，与开发板结构对齐）
        safe_output_session = _re.sub(r'[^\w\-]', '_', str(session_name))
        safe_output_orig = _re.sub(r'[^\w\.\-]', '_', str(orig_name))
        output_session_dir = os.path.join(OUTPUT_DIR, safe_output_session)
        os.makedirs(output_session_dir, exist_ok=True)
        output_path = os.path.join(output_session_dir, safe_output_orig)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(content)
        os.replace(tmp_path, output_path)  # 原子写入

        print(f"[TBJU] 同步文件已接收: {safe_name} ({len(content)} bytes)")
        print(f"[TBJU] 已保存到: {output_path}")
        return JSONResponse(content={"ok": True, "filename": safe_name, "size": len(content), "output_path": output_path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/api/files/list")
async def list_synced_files(device_id: Optional[str] = Query(default=None)):
    """列出已同步的文件"""
    files = []
    if not os.path.exists(SYNCED_DIR):
        return JSONResponse(content={"ok": True, "files": []})

    for dev_dir in sorted(os.listdir(SYNCED_DIR), reverse=True):
        if device_id and dev_dir != device_id:
            continue
        dev_path = os.path.join(SYNCED_DIR, dev_dir)
        if not os.path.isdir(dev_path):
            continue
        for date_dir in sorted(os.listdir(dev_path), reverse=True):
            date_path = os.path.join(dev_path, date_dir)
            if not os.path.isdir(date_path):
                continue
            for fname in sorted(os.listdir(date_path), reverse=True):
                fpath = os.path.join(date_path, fname)
                files.append({
                    "device_id": dev_dir,
                    "date": date_dir,
                    "filename": fname,
                    "size": os.path.getsize(fpath),
                    "path": f"{dev_dir}/{date_dir}/{fname}",
                })

    return JSONResponse(content={"ok": True, "files": files})


@app.get("/api/files/download/{path:path}")
async def download_synced_file(path: str):
    """下载同步的文件"""
    # 防止路径穿越
    full_path = os.path.normpath(os.path.join(SYNCED_DIR, path))
    synced_root = os.path.normpath(SYNCED_DIR)
    if not full_path.startswith(synced_root + os.sep) and full_path != synced_root:
        raise HTTPException(status_code=403, detail="access denied")
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="file not found")

    from fastapi.responses import FileResponse
    return FileResponse(full_path, filename=os.path.basename(full_path), media_type="text/csv")


@app.get("/api/export/csv")
async def export_csv(event_type: Optional[str] = Query(default=None)):
    """
    导出事件为 CSV

    参数:
        event_type: 事件类型筛选（可选）
    """
    events = db.get_events_for_export(event_type=event_type)

    output = io.StringIO()
    writer = csv.writer(output)

    # 写入表头
    headers = [
        "event_id", "device_id", "board_ip", "event_type", "severity",
        "message", "source", "frame", "ocr_texts", "class_counts",
        "created_at", "received_at"
    ]
    writer.writerow(headers)

    # 写入数据
    for event in events:
        writer.writerow([
            event.get("event_id", ""),
            event.get("device_id", ""),
            event.get("board_ip", ""),
            event.get("event_type", ""),
            event.get("severity", ""),
            event.get("message", ""),
            event.get("source", ""),
            event.get("frame", ""),
            event.get("ocr_texts", ""),
            event.get("class_counts", ""),
            event.get("created_at", ""),
            event.get("received_at", "")
        ])

    output.seek(0)
    filename = f"tbju_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.delete("/api/events/test")
async def delete_test_events():
    """删除所有测试事件"""
    count = db.clear_test_events()
    return JSONResponse(content={"ok": True, "deleted": count})


@app.delete("/api/events/all")
async def delete_all_events():
    """删除所有事件"""
    count = db.clear_all_events()
    return JSONResponse(content={"ok": True, "deleted": count})


# ============ 远程控制命令接口 ============

@app.post("/api/commands")
async def create_command(request: Request):
    data = await request.json()
    if not data or 'action' not in data:
        return JSONResponse(status_code=400, content={'error': '缺少 action'})
    try:
        cmd = db.insert_command(data['action'], data.get('params'), data.get('device_id', 'ELF2-TBJU-01'))
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})
    return JSONResponse(status_code=201, content=cmd)


@app.get("/api/commands/pending")
async def get_pending_commands(device_id: str = Query(default='ELF2-TBJU-01'), limit: int = Query(default=5)):
    db.recover_stale_commands()
    commands = db.pick_pending_commands(device_id, limit)
    return JSONResponse(content={'commands': commands})


@app.post("/api/commands/{command_id}/ack")
async def acknowledge_command(command_id: str, request: Request):
    data = await request.json()
    if not data:
        return JSONResponse(status_code=400, content={'error': '缺少请求体'})
    ok = db.ack_command(command_id, data.get('success', False), data.get('message', ''))
    if not ok:
        return JSONResponse(status_code=404, content={'error': '命令不存在或状态不正确'})
    return JSONResponse(content={'status': 'ok'})


@app.get("/api/commands/history")
async def command_history(device_id: str = Query(default='ELF2-TBJU-01'), limit: int = Query(default=20)):
    history = db.get_command_history(device_id, limit)
    return JSONResponse(content={'history': history})


@app.post("/api/commands/heartbeat")
async def device_heartbeat(request: Request):
    data = await request.json()
    device_id = data.get('device_id', 'ELF2-TBJU-01')
    device_status[device_id] = {
        'ip': data.get('ip', 'unknown'),
        'status': data.get('status', 'idle'),
        'last_seen': datetime.now().isoformat(),
        'version': data.get('version', ''),
        'commands_executed': data.get('commands_executed', 0),
    }
    return JSONResponse(content={'status': 'ok'})


@app.get("/api/devices/status")
async def get_device_status():
    return JSONResponse(content=device_status)


# ============ 测试接口 ============

@app.post("/api/test/send-sample")
async def send_sample_event():
    """
    发送一个示例事件（用于测试看板是否正常工作）

    访问: POST http://localhost:8000/api/test/send-sample
    """
    import random

    sample_events = [
        {
            "event_id": f"test-{int(datetime.now().timestamp()*1000)}",
            "device_id": "ELF2-TBJU-01",
            "board_ip": "sample-board",
            "event_type": "debris_alarm",
            "severity": "critical",
            "message": "发现轨道侵入异物",
            "source": "camera",
            "frame": random.randint(100, 9999),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detections": [
                {
                    "class_id": 4,
                    "class_name": "track_intrusion_debris",
                    "bbox": [120, 80, 260, 190],
                    "confidence": round(random.uniform(0.7, 0.99), 4),
                    "ocr_text": ""
                }
            ],
            "class_counts": {"track_intrusion_debris": 1, "track_region": 1},
            "ocr_texts": [],
            "timing": {
                "yolo_ms": round(random.uniform(30, 60), 1),
                "ocr_ms": 0,
                "total_ms": round(random.uniform(30, 60), 1),
                "fps": round(random.uniform(8, 15), 1)
            },
            "system": {
                "cpu_percent": round(random.uniform(30, 80), 1),
                "memory_percent": round(random.uniform(30, 70), 1),
                "memory_used_mb": round(random.uniform(2000, 5000), 1),
                "process_rss_mb": round(random.uniform(300, 600), 1),
                "max_temp_c": round(random.uniform(45, 75), 1),
                "npu_load_percent": round(random.uniform(20, 60), 1),
                "gpu_load_percent": None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "thumbnail": None
        },
        {
            "event_id": f"test-{int(datetime.now().timestamp()*1000)}-ocr",
            "device_id": "ELF2-TBJU-01",
            "board_ip": "sample-board",
            "event_type": "ocr_record",
            "severity": "info",
            "message": "识别到车号区域",
            "source": "camera",
            "frame": random.randint(100, 9999),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ocr_texts": [f"TBJU{random.randint(1000000, 9999999)}"],
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "TBJU_region",
                    "bbox": [88, 54, 280, 102],
                    "confidence": round(random.uniform(0.7, 0.99), 4),
                    "ocr_text": f"TBJU{random.randint(1000000, 9999999)}"
                }
            ],
            "timing": {
                "yolo_ms": round(random.uniform(30, 50), 1),
                "ocr_ms": round(random.uniform(5, 15), 1),
                "total_ms": round(random.uniform(35, 65), 1),
                "fps": round(random.uniform(8, 15), 1)
            },
            "system": {
                "cpu_percent": round(random.uniform(30, 60), 1),
                "memory_percent": round(random.uniform(30, 50), 1),
                "max_temp_c": round(random.uniform(40, 65), 1),
                "npu_load_percent": round(random.uniform(20, 50), 1)
            },
            "thumbnail": None
        },
        {
            "event_id": f"test-{int(datetime.now().timestamp()*1000)}-sys",
            "device_id": "ELF2-TBJU-01",
            "board_ip": "sample-board",
            "event_type": "system_alarm",
            "alarm_key": "temp_high",
            "severity": "warning",
            "message": f"警告，系统温度偏高，当前 {random.randint(70, 85)} 度",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system": {
                "cpu_percent": round(random.uniform(80, 99), 1),
                "memory_percent": round(random.uniform(50, 80), 1),
                "max_temp_c": round(random.uniform(70, 85), 1)
            }
        },
        {
            "event_id": f"test-{int(datetime.now().timestamp()*1000)}-net",
            "device_id": "ELF2-TBJU-01",
            "event_type": "network_test",
            "severity": "info",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "board_ip": "sample-board",
            "message": "ELF2 board network upload test"
        }
    ]

    event = random.choice(sample_events)
    event_id = db.insert_event(event)

    return JSONResponse(content={
        "ok": True,
        "event_id": event_id,
        "event_type": event["event_type"],
        "message": f"示例 {event['event_type']} 事件已发送"
    })


if __name__ == "__main__":
    import uvicorn
    lan_ips = get_lan_ipv4s()
    print("=" * 60)
    print("  TBJU 轨道异物智能检测远程看板")
    print("=" * 60)
    print(f"  看板页面: http://localhost:{PORT}")
    if lan_ips:
        print("  如果出现多个地址，请选择与 ELF2 板端 IP 同网段的那个。")
        for ip in lan_ips:
            print(f"  局域网看板: http://{ip}:{PORT}")
            print(f"  板端上传: http://{ip}:{PORT}/api/events")
    else:
        print(f"  板端上传: http://电脑端IP:{PORT}/api/events")
    print(f"  测试接口: POST http://localhost:{PORT}/api/test/send-sample")
    print("=" * 60)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
