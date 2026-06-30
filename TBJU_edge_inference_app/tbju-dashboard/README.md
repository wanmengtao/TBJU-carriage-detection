# 远程看板 (tbju-dashboard)

FastAPI Web 应用，用于接收开发板上传的检测事件和监控数据。

---

## 功能概述

| Tab | 名称 | 功能 |
|-----|------|------|
| Tab1 | 总览 | KPI 卡片 + 告警 + 远程控制 + 日志/命令历史 + 最近事件 + 设备状态 |
| Tab2 | 检测 | 同步文件列表 + 事件表格（7 列）+ 操作栏 |
| Tab3 | 性能 | 6 数值卡片（CPU/内存/温度/推理耗时/NPU/GPU）+ 7 图表 |

---

## 目录结构

```
tbju-dashboard/
├── app.py                             # FastAPI 主应用
├── database.py                        # SQLite 数据库操作
├── requirements.txt                   # Python 依赖
├── README.md                          # 本文件
└── static/                            # 前端静态文件
    ├── index.html                     # HTML 页面
    ├── script.js                      # JavaScript 逻辑
    └── style.css                      # 样式（深色工业风）
```

---

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py

# 访问 http://localhost:8000
```

---

## API 接口

### 事件上传

```http
POST /api/events
Content-Type: application/json

{
  "timestamp": "2026-06-30T12:00:00",
  "event_type": "debris",
  "confidence": 0.95,
  "bbox": [100, 100, 200, 200],
  "image_path": "output/images/session/001_result.jpg"
}
```

### 文件上传

```http
POST /api/files/upload
Content-Type: multipart/form-data

file: result.csv
source_dir: camera
```

### 获取事件列表

```http
GET /api/events?limit=100&offset=0
```

### 远程命令

```http
GET /api/commands/poll
POST /api/commands/ack
```

---

## 安全特性

- 路径穿越防护：`os.path.normpath` + 前缀校验
- 文件名正则清理：`[^\w\-]` / `[^\w\.\-]`
- SQLite `BEGIN IMMEDIATE` 防竞态
- `PRAGMA busy_timeout=5000` 防并发写入冲突

---

## 使用方式

1. 在 Windows PC 上启动看板服务
2. 在开发板 GUI 中填写看板地址（如 `http://PC端IP:8000/api/events`）
3. 勾选"启用上传"，点击"测试连接"
4. 开发板检测到事件时自动上传到看板

---

## 依赖

```
fastapi>=0.109.0
uvicorn>=0.27.0
python-multipart>=0.0.6
```
