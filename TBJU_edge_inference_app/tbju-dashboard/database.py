"""
TBJU 远程看板 - SQLite 数据库模块
"""
import sqlite3
import json
import os
import uuid
from datetime import datetime, timedelta
from contextlib import contextmanager

ALLOWED_ACTIONS = {
    'open_camera', 'start_detection', 'stop_detection',
    'pause_detection', 'start_capacity', 'stop_capacity',
    'toggle_recording', 'stop_recording', 'show_status',
    'mute_alarm', 'unmute_alarm', 'set_param'
}

DB_PATH = os.path.join(os.path.dirname(__file__), "tbju_events.db")


def get_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """数据库上下文管理器"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """初始化数据库表"""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE,
                device_id TEXT,
                board_ip TEXT,
                event_type TEXT,
                severity TEXT,
                message TEXT,
                source TEXT,
                frame INTEGER,
                ocr_texts TEXT,
                class_counts TEXT,
                detections TEXT,
                timing TEXT,
                system TEXT,
                thumbnail_path TEXT,
                raw_json TEXT NOT NULL,
                created_at TEXT,
                received_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_device_id ON events(device_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity)
        """)
        # 远程控制命令队列
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command_id TEXT UNIQUE NOT NULL,
                device_id TEXT NOT NULL DEFAULT 'ELF2-TBJU-01',
                action TEXT NOT NULL,
                params TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT,
                created_at TEXT NOT NULL,
                picked_at TEXT,
                acked_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_commands_status
            ON commands(device_id, status, created_at)
        """)
    print(f"[DB] 数据库初始化完成: {DB_PATH}")


def insert_event(event_data: dict, thumbnail_path: str = None) -> str:
    """
    插入事件记录

    Args:
        event_data: 事件 JSON 数据
        thumbnail_path: 缩略图文件路径（可选）

    Returns:
        event_id
    """
    event_id = event_data.get("event_id", f"auto-{int(datetime.now().timestamp()*1000)}")
    received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO events (
                event_id, device_id, board_ip, event_type, severity,
                message, source, frame, ocr_texts, class_counts,
                detections, timing, system, thumbnail_path,
                raw_json, created_at, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id,
            event_data.get("device_id"),
            event_data.get("board_ip"),
            event_data.get("event_type"),
            event_data.get("severity"),
            event_data.get("message"),
            event_data.get("source"),
            event_data.get("frame"),
            json.dumps(event_data.get("ocr_texts", []), ensure_ascii=False),
            json.dumps(event_data.get("class_counts", {}), ensure_ascii=False),
            json.dumps(event_data.get("detections", []), ensure_ascii=False),
            json.dumps(event_data.get("timing", {}), ensure_ascii=False),
            json.dumps(event_data.get("system", {}), ensure_ascii=False),
            thumbnail_path,
            json.dumps(event_data, ensure_ascii=False),
            event_data.get("created_at"),
            received_at
        ))

    return event_id


def get_recent_events(limit: int = 50, event_type: str = None) -> list:
    """
    获取最近事件

    Args:
        limit: 返回数量限制
        event_type: 事件类型筛选（可选）

    Returns:
        事件列表
    """
    with get_db() as conn:
        if event_type:
            rows = conn.execute("""
                SELECT * FROM events
                WHERE event_type = ?
                ORDER BY id DESC
                LIMIT ?
            """, (event_type, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM events
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

    return [dict(row) for row in rows]


def get_event_by_id(event_id: str) -> dict:
    """
    根据 event_id 获取事件详情

    Args:
        event_id: 事件 ID

    Returns:
        事件详情字典，未找到返回 None
    """
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM events WHERE event_id = ?
        """, (event_id,)).fetchone()

    return dict(row) if row else None


def get_stats() -> dict:
    """
    获取统计数据

    Returns:
        统计信息字典
    """
    with get_db() as conn:
        # 总事件数
        total = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()["cnt"]

        # 各类型事件数
        debris_alarm = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE event_type = 'debris_alarm'"
        ).fetchone()["cnt"]

        ocr_record = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE event_type = 'ocr_record'"
        ).fetchone()["cnt"]

        system_alarm = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE event_type = 'system_alarm'"
        ).fetchone()["cnt"]

        network_test = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE event_type = 'network_test'"
        ).fetchone()["cnt"]

        # 最后事件时间和设备
        last_event = conn.execute("""
            SELECT created_at, board_ip FROM events
            ORDER BY id DESC LIMIT 1
        """).fetchone()

        # 设备数量
        device_count = conn.execute(
            "SELECT COUNT(DISTINCT device_id) as cnt FROM events WHERE device_id IS NOT NULL"
        ).fetchone()["cnt"]

    return {
        "ok": True,
        "device_count": device_count,
        "total_events": total,
        "debris_alarm_count": debris_alarm,
        "ocr_record_count": ocr_record,
        "system_alarm_count": system_alarm,
        "network_test_count": network_test,
        "last_event_at": last_event["created_at"] if last_event else None,
        "last_device_ip": last_event["board_ip"] if last_event else None
    }


def clear_test_events() -> int:
    """删除所有测试事件，返回删除数量"""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM events WHERE event_type = 'network_test'")
        return cursor.rowcount


def clear_all_events() -> int:
    """删除所有事件，返回删除数量"""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM events")
        return cursor.rowcount


def get_events_for_export(event_type: str = None) -> list:
    """导出事件数据"""
    with get_db() as conn:
        if event_type:
            rows = conn.execute("""
                SELECT * FROM events WHERE event_type = ? ORDER BY id DESC
            """, (event_type,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM events ORDER BY id DESC
            """).fetchall()

    return [dict(row) for row in rows]


# ============ 远程控制命令队列 ============

def insert_command(action: str, params: dict = None, device_id: str = 'ELF2-TBJU-01') -> dict:
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"不允许的动作: {action}")
    command_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    params_json = json.dumps(params, ensure_ascii=False) if params else None
    with get_db() as conn:
        conn.execute(
            'INSERT INTO commands (command_id,device_id,action,params,status,created_at) VALUES (?,?,?,?,?,?)',
            (command_id, device_id, action, params_json, 'pending', created_at)
        )
    return {'command_id': command_id, 'device_id': device_id, 'action': action, 'params': params, 'status': 'pending', 'created_at': created_at}


def pick_pending_commands(device_id: str = 'ELF2-TBJU-01', limit: int = 5) -> list:
    with get_db() as conn:
        now = datetime.now().isoformat()
        # BEGIN IMMEDIATE 获取写锁，防止并发 SELECT 竞态
        conn.execute('BEGIN IMMEDIATE')
        rows = conn.execute(
            'SELECT id,command_id,action,params,created_at FROM commands WHERE device_id=? AND status=? ORDER BY created_at ASC LIMIT ?',
            (device_id, 'pending', limit)
        ).fetchall()
        if not rows:
            return []
        ids = [row['id'] for row in rows]
        ph = ','.join('?' * len(ids))
        conn.execute(f'UPDATE commands SET status=?,picked_at=? WHERE id IN ({ph})', ['picked', now] + ids)
    return [{'command_id': r['command_id'], 'action': r['action'], 'params': json.loads(r['params']) if r['params'] else None, 'created_at': r['created_at']} for r in rows]


def ack_command(command_id: str, success: bool, message: str = '') -> bool:
    now = datetime.now().isoformat()
    result_json = json.dumps({'success': success, 'message': message}, ensure_ascii=False)
    with get_db() as conn:
        cursor = conn.execute(
            'UPDATE commands SET status=?,result=?,acked_at=? WHERE command_id=? AND status=?',
            ('done' if success else 'failed', result_json, now, command_id, 'picked')
        )
        return cursor.rowcount > 0


def get_command_history(device_id: str = 'ELF2-TBJU-01', limit: int = 20) -> list:
    with get_db() as conn:
        rows = conn.execute(
            'SELECT command_id,action,params,status,result,created_at,picked_at,acked_at FROM commands WHERE device_id=? ORDER BY created_at DESC LIMIT ?',
            (device_id, limit)
        ).fetchall()
    return [{'command_id': r['command_id'], 'action': r['action'], 'params': json.loads(r['params']) if r['params'] else None, 'status': r['status'], 'result': json.loads(r['result']) if r['result'] else None, 'created_at': r['created_at'], 'picked_at': r['picked_at'], 'acked_at': r['acked_at']} for r in rows]


def cleanup_old_commands(days: int = 7):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with get_db() as conn:
        conn.execute("DELETE FROM commands WHERE created_at<? AND status IN ('done','failed','timeout')", (cutoff,))


def recover_stale_commands(timeout_minutes: int = 5, max_retries: int = 2):
    cutoff = (datetime.now() - timedelta(minutes=timeout_minutes)).isoformat()
    with get_db() as conn:
        # 超过重试次数的标记为 timeout
        conn.execute(
            "UPDATE commands SET status='timeout' WHERE status='picked' AND picked_at<? "
            "AND COALESCE(json_extract(result,'$.retry_count'),0)>=?",
            (cutoff, max_retries)
        )
        # 其余恢复为 pending，重试次数 +1
        conn.execute(
            "UPDATE commands SET status='pending', picked_at=NULL, "
            "result=json_set(COALESCE(result,'{}'),'$.retry_count',"
            "COALESCE(json_extract(result,'$.retry_count'),0)+1) "
            "WHERE status='picked' AND picked_at<? "
            "AND COALESCE(json_extract(result,'$.retry_count'),0)<?",
            (cutoff, max_retries)
        )
