/**
 * TBJU 远程看板 v2 - 三 Tab 布局
 */

// ===== 全局状态 =====
const state = {
    events: [],
    stats: null,
    lastEventTime: null,
    lastEventTimestamp: null,
    currentFilter: '',
    currentTab: 'overview',
    bandwidth: { data: [], peakKbps: 0, totalKb: 0 },
};

// ===== 常量 =====
const POLL_INTERVAL = 1000;
const MAX_CHART_POINTS = 30;
const EVENT_TYPE_NAMES = {
    'debris_alarm': '异物告警',
    'ocr_record': 'OCR记录',
    'system_alarm': '系统告警',
    'system_metric': '性能指标',
    'network_test': '测试事件',
    'app_log': '日志',
};
const SEVERITY_NAMES = { 'critical': '严重', 'warning': '警告', 'info': '信息' };

// ===== 初始化 =====
document.addEventListener('DOMContentLoaded', () => {
    installZoomGuard();
    initCharts();
    bindTabs();
    bindEvents();

    // 首次加载
    refreshEvents();
    refreshStats();
    refreshBandwidth();

    // 轮询
    setInterval(pollTick, POLL_INTERVAL);
    setInterval(refreshStats, POLL_INTERVAL * 5);

    // 时钟
    updateClock();
    setInterval(updateClock, 1000);

    // 设备状态
    setInterval(checkDeviceStatus, 1000);

    // 远程控制
    refreshHistory();
    refreshDeviceStatus();
    refreshLogs();
    setInterval(() => { refreshHistory(); refreshDeviceStatus(); refreshLogs(); }, 3000);
});

// ===== Tab 切换 =====
function bindTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            switchTab(btn.dataset.tab);
        });
    });
}

function switchTab(tabName) {
    state.currentTab = tabName;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.tab-btn[data-tab="${tabName}"]`).classList.add('active');
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');
    localStorage.setItem('tbju_tab', tabName);
}

// 恢复上次 Tab
(function restoreTab() {
    const saved = localStorage.getItem('tbju_tab');
    if (saved && document.getElementById(`tab-${saved}`)) {
        setTimeout(() => switchTab(saved), 0);
    }
})();

// ===== 轮询策略 =====
function pollTick() {
    refreshEvents();
    if (state.currentTab === 'performance') {
        refreshBandwidth();
    }
}

// ===== 缩放保护 =====
function installZoomGuard() {
    window.addEventListener('keydown', (e) => {
        if (!(e.ctrlKey || e.metaKey)) return;
        if (['+', '-', '=', '0'].includes(e.key)) e.preventDefault();
    });
}

// ===== 绑定事件 =====
function bindEvents() {
    document.getElementById('eventTypeFilter').addEventListener('change', (e) => {
        state.currentFilter = e.target.value;
        renderEventTable();
    });
}

// ===== 时钟 =====
function updateClock() {
    document.getElementById('currentTime').textContent =
        new Date().toLocaleTimeString('zh-CN', { hour12: false });
}

// ===== 设备状态 =====
function checkDeviceStatus() {
    const el = document.getElementById('deviceStatus');
    const text = document.getElementById('deviceStatusText');
    if (!state.lastEventTimestamp) {
        el.className = 'status-value';
        text.textContent = '等待连接';
        return;
    }
    const diff = Date.now() - state.lastEventTimestamp;
    if (diff < 10000) { el.className = 'status-value online'; text.textContent = '在线'; }
    else if (diff < 60000) { el.className = 'status-value delay'; text.textContent = '延迟'; }
    else { el.className = 'status-value offline'; text.textContent = '离线'; }
}

// ===== 刷新事件 =====
async function refreshEvents() {
    try {
        const resp = await fetch('/api/events/recent?limit=100');
        const data = await resp.json();
        if (data.ok) {
            state.events = data.events;
            if (state.events.length > 0) {
                const last = state.events[0];
                state.lastEventTime = last.created_at || last.received_at;
                state.lastEventTimestamp = new Date(state.lastEventTime).getTime() || Date.now();
                document.getElementById('ovLastEvent').textContent = formatTime(state.lastEventTime, true);
                document.getElementById('ovBoardIp').textContent = last.board_ip || '--';
            }
            renderEventTable();
            renderRecentEvents();
            updateKPIs();
            updateAlertSection();
            updateChartData();
            updateDeviceCard();
        }
    } catch (e) { console.error('[TBJU] 获取事件失败:', e); }
}

// ===== 刷新统计 =====
async function refreshStats() {
    try {
        const resp = await fetch('/api/stats');
        const data = await resp.json();
        if (data.ok) { state.stats = data; updateKPIs(); }
    } catch (e) {}
}

// ===== 刷新带宽 =====
async function refreshBandwidth() {
    try {
        const resp = await fetch('/api/bandwidth');
        const data = await resp.json();
        if (data.ok) {
            state.bandwidth.data.push(data.kb_per_sec);
            if (state.bandwidth.data.length > MAX_CHART_POINTS) state.bandwidth.data.shift();
            state.bandwidth.peakKbps = data.peak_kb_per_sec;
            state.bandwidth.totalKb = data.total_kb;

            drawChart('bandwidthChart', {
                datasets: [{ label: 'KB/s', data: state.bandwidth.data, color: '#00b4d8' }]
            });

            const statsEl = document.getElementById('bwStats');
            if (statsEl) {
                statsEl.textContent =
                    `当前: ${data.kb_per_sec.toFixed(1)} KB/s | ` +
                    `峰值: ${data.peak_kb_per_sec.toFixed(1)} KB/s | ` +
                    `累计: ${data.total_mb.toFixed(2)} MB`;
            }

            // Tab3 数值卡片
            const pfBw = document.getElementById('pfBw');
            if (pfBw) pfBw.textContent = data.kb_per_sec.toFixed(1) + ' KB/s';

            // Tab1 设备状态
            const ovBw = document.getElementById('ovBw');
            if (ovBw) ovBw.textContent = data.kb_per_sec.toFixed(1) + ' KB/s';
        }
    } catch (e) {}
}

// ===== KPI 卡片 =====
function updateKPIs() {
    if (state.stats) {
        document.getElementById('kpiTotalEvents').textContent = state.stats.total_events || 0;
        document.getElementById('kpiDebrisAlarm').textContent = state.stats.debris_alarm_count || 0;
        document.getElementById('kpiOcrRecord').textContent = state.stats.ocr_record_count || 0;
        document.getElementById('kpiSystemAlarm').textContent = state.stats.system_alarm_count || 0;
    }
    if (state.events.length > 0) {
        const latest = state.events.find(e => e.event_type === 'system_metric') || state.events.find(e => e.system);
        if (latest && latest.system) {
            const sys = typeof latest.system === 'string' ? JSON.parse(latest.system) : latest.system;
            // Tab3 性能卡片
            const pfCpu = document.getElementById('pfCpu');
            const pfMem = document.getElementById('pfMem');
            const pfTemp = document.getElementById('pfTemp');
            const pfTiming = document.getElementById('pfTiming');
            if (pfCpu && sys.cpu_percent != null) pfCpu.textContent = sys.cpu_percent.toFixed(1) + '%';
            if (pfMem && sys.memory_percent != null) pfMem.textContent = sys.memory_percent.toFixed(1) + '%';
            if (pfTemp && sys.max_temp_c != null) pfTemp.textContent = sys.max_temp_c.toFixed(1) + '°C';
            // 推理耗时
            const timing = typeof latest.timing === 'string' ? JSON.parse(latest.timing) : latest.timing;
            if (pfTiming && timing && timing.total_ms != null) pfTiming.textContent = timing.total_ms.toFixed(1) + 'ms';
        }
    }
}

// ===== Tab1: 设备状态卡片 =====
function updateDeviceCard() {
    if (state.events.length === 0) return;
    const latest = state.events.find(e => e.event_type === 'system_metric') || state.events.find(e => e.system);
    if (latest && latest.system) {
        const sys = typeof latest.system === 'string' ? JSON.parse(latest.system) : latest.system;
        if (sys.cpu_percent != null) document.getElementById('ovCpu').textContent = sys.cpu_percent.toFixed(0) + '%';
        if (sys.memory_percent != null) document.getElementById('ovMem').textContent = sys.memory_percent.toFixed(0) + '%';
        if (sys.max_temp_c != null) document.getElementById('ovTemp').textContent = sys.max_temp_c.toFixed(1) + '°';
        if (sys.gpu_load_percent != null) document.getElementById('ovGpu').textContent = sys.gpu_load_percent.toFixed(0) + '%';
        else document.getElementById('ovGpu').textContent = 'N/A';
    }
    // 上传链路状态
    const linkEl = document.getElementById('ovLinkStatus');
    if (state.lastEventTimestamp) {
        const diff = Date.now() - state.lastEventTimestamp;
        if (diff < 15000) {
            linkEl.innerHTML = '<span class="dot online-dot"></span> 正常';
        } else if (diff < 60000) {
            linkEl.innerHTML = '<span class="dot delay-dot"></span> 延迟';
        } else {
            linkEl.innerHTML = '<span class="dot offline-dot"></span> 断开';
        }
    }
}

// ===== Tab1: 最近事件摘要 =====
function renderRecentEvents() {
    const container = document.getElementById('recentEventsList');
    if (state.events.length === 0) {
        container.innerHTML = '<p class="empty-hint">暂无事件</p>';
        return;
    }
    const recent = state.events.slice(0, 8);
    container.innerHTML = recent.map(ev => {
        const typeClass = `type-${ev.event_type}`;
        const typeName = EVENT_TYPE_NAMES[ev.event_type] || ev.event_type;
        return `<div class="recent-event-item" onclick="switchTab('detection')">
            <span class="re-time">${formatTime(ev.created_at, true)}</span>
            <span class="re-type type-badge ${typeClass}">${typeName}</span>
            <span class="re-msg">${escapeHtml(ev.message || '-')}</span>
        </div>`;
    }).join('');
}

// ===== 告警区 =====
function updateAlertSection() {
    const section = document.getElementById('alertSection');
    const critical = state.events.find(e =>
        e.event_type === 'debris_alarm' || e.severity === 'critical' ||
        (e.event_type === 'system_alarm' && e.severity === 'warning')
    );
    if (!critical) { section.style.display = 'none'; return; }
    section.style.display = 'block';
    document.getElementById('alertTime').textContent = critical.created_at || '';

    const content = document.getElementById('alertContent');
    let html = '';
    html += alertItem('事件类型', EVENT_TYPE_NAMES[critical.event_type] || critical.event_type, critical.severity);
    html += alertItem('严重等级', SEVERITY_NAMES[critical.severity] || critical.severity, critical.severity);
    html += alertItem('发生时间', critical.created_at || '-');
    if (critical.frame) html += alertItem('帧号', critical.frame);
    if (critical.class_counts) {
        const counts = typeof critical.class_counts === 'string' ? JSON.parse(critical.class_counts) : critical.class_counts;
        html += alertItem('类别', Object.keys(counts).map(k => `${k}(${counts[k]})`).join(', '));
    }
    // 异物告警时显示识别帧图片
    if (critical.event_type === 'debris_alarm' && critical.thumbnail_path) {
        html += `<div class="alert-item alert-image" style="grid-column: span 2;">
            <div class="label">识别帧</div>
            <div class="value"><img src="/${critical.thumbnail_path}" style="max-width:100%;max-height:200px;border-radius:6px;border:1px solid var(--border-color);" /></div>
        </div>`;
    } else if (critical.thumbnail_path) {
        html += `<div class="alert-item" style="grid-column: span 2;">
            <div class="label">缩略图</div>
            <div class="value"><img src="/${critical.thumbnail_path}" style="max-width:200px;max-height:100px;border-radius:4px;" /></div>
        </div>`;
    }
    content.innerHTML = html;
}

function alertItem(label, value, severity) {
    const cls = severity ? ` ${severity}` : '';
    return `<div class="alert-item"><div class="label">${label}</div><div class="value${cls}">${value || '-'}</div></div>`;
}

// ===== Tab2: 事件表格（7 列） =====
function renderEventTable() {
    const tbody = document.getElementById('eventTableBody');
    const empty = document.getElementById('eventEmpty');
    let filtered = state.events;
    if (state.currentFilter) filtered = state.events.filter(e => e.event_type === state.currentFilter);

    if (filtered.length === 0) { tbody.innerHTML = ''; empty.classList.add('show'); return; }
    empty.classList.remove('show');

    tbody.innerHTML = filtered.map(ev => {
        const isCrit = ev.severity === 'critical' || ev.event_type === 'debris_alarm';
        const timing = parseJsonField(ev.timing);
        const system = parseJsonField(ev.system);
        const detections = parseJsonField(ev.detections);
        const ocrTexts = parseJsonField(ev.ocr_texts);

        let ocrStr = '-';
        if (ocrTexts && ocrTexts.length > 0) ocrStr = ocrTexts.join(', ');
        else if (detections && detections.length > 0) { const d = detections.find(d => d.ocr_text); if (d) ocrStr = d.ocr_text; }

        let thumb = '<span class="no-thumbnail">-</span>';
        if (ev.thumbnail_path) thumb = `<img class="thumbnail-mini" src="/${ev.thumbnail_path}" onerror="this.outerHTML='-'" />`;

        return `<tr class="${isCrit ? 'critical-row' : ''}" onclick="showEventDetail('${ev.event_id}')">
            <td>${formatTime(ev.created_at)}</td>
            <td><span class="severity-badge severity-${ev.severity || 'info'}">${SEVERITY_NAMES[ev.severity] || ev.severity || '-'}</span></td>
            <td><span class="type-badge type-${ev.event_type}">${EVENT_TYPE_NAMES[ev.event_type] || ev.event_type}</span></td>
            <td title="${escapeHtml(ocrStr)}">${ocrStr}</td>
            <td>${timing ? (timing.fps ? timing.fps.toFixed(1) : '-') : '-'}</td>
            <td>${system ? (system.max_temp_c ? system.max_temp_c.toFixed(1) + '°C' : '-') : '-'}</td>
            <td>${thumb}</td>
        </tr>`;
    }).join('');
}

// ===== 事件详情模态框 =====
async function showEventDetail(eventId) {
    try {
        const resp = await fetch(`/api/events/${eventId}`);
        const data = await resp.json();
        if (!data.ok) { showToast('error', '错误', '获取事件详情失败'); return; }

        const ev = data.event;
        const timing = parseJsonField(ev.timing);
        const system = parseJsonField(ev.system);
        const detections = parseJsonField(ev.detections);
        const ocrTexts = parseJsonField(ev.ocr_texts);

        let html = '<div class="detail-grid">';
        html += detailItem('事件ID', ev.event_id, true);
        html += detailItem('事件类型', EVENT_TYPE_NAMES[ev.event_type] || ev.event_type);
        html += detailItem('严重等级', SEVERITY_NAMES[ev.severity] || ev.severity);
        html += detailItem('设备ID', ev.device_id);
        html += detailItem('板端IP', ev.board_ip);
        html += detailItem('来源', ev.source);
        html += detailItem('帧号', ev.frame);
        html += detailItem('消息', ev.message);
        html += detailItem('创建时间', ev.created_at);
        html += '</div>';

        if (detections && detections.length > 0) {
            html += '<div class="detail-section"><h3>🎯 检测框</h3><div class="detection-list">';
            detections.forEach(d => {
                html += `<div class="detection-item">
                    <span class="detection-class">${d.class_name || '?'}</span>
                    <span class="detection-confidence">${(d.confidence * 100).toFixed(1)}%</span>
                    <span class="detection-bbox">[${d.bbox ? d.bbox.join(', ') : '-'}]</span>
                    ${d.ocr_text ? `<span class="detection-ocr">OCR: ${d.ocr_text}</span>` : ''}
                </div>`;
            });
            html += '</div></div>';
        }

        if (ocrTexts && ocrTexts.length > 0) {
            html += `<div class="detail-section"><h3>🔤 OCR</h3><div style="color:var(--accent-yellow);font-family:var(--font-mono);font-size:16px;">${ocrTexts.join(', ')}</div></div>`;
        }

        if (system && Object.keys(system).length > 0) {
            html += '<div class="detail-section"><h3>📊 系统指标</h3><div class="detail-grid">';
            if (system.cpu_percent != null) html += detailItem('CPU', system.cpu_percent + '%');
            if (system.memory_percent != null) html += detailItem('内存', system.memory_percent + '%');
            if (system.max_temp_c != null) html += detailItem('温度', system.max_temp_c + '°C');
            if (system.npu_load_percent != null) html += detailItem('NPU', system.npu_load_percent + '%');
            html += '</div></div>';
        }

        if (timing && Object.keys(timing).length > 0) {
            html += '<div class="detail-section"><h3>⏱️ 推理耗时</h3><div class="detail-grid">';
            if (timing.yolo_ms != null) html += detailItem('YOLO', timing.yolo_ms + ' ms');
            if (timing.ocr_ms != null) html += detailItem('OCR', timing.ocr_ms + ' ms');
            if (timing.total_ms != null) html += detailItem('总计', timing.total_ms + ' ms');
            if (timing.fps != null) html += detailItem('FPS', timing.fps.toFixed(1));
            html += '</div></div>';
        }

        if (ev.thumbnail_path) {
            html += `<div class="detail-section"><h3>🖼️ 缩略图</h3><img class="thumbnail-preview" src="/${ev.thumbnail_path}" /></div>`;
        }

        html += `<div class="detail-section"><h3>📄 原始 JSON</h3><div class="json-viewer">${JSON.stringify(ev, null, 2)}</div></div>`;

        document.getElementById('modalBody').innerHTML = html;
        document.getElementById('eventModal').style.display = 'flex';
    } catch (e) {
        showToast('error', '错误', '获取事件详情失败');
    }
}

function detailItem(label, value, highlight) {
    return `<div class="detail-item"><div class="label">${label}</div><div class="value${highlight ? ' highlight' : ''}">${value || '-'}</div></div>`;
}

function closeModal() { document.getElementById('eventModal').style.display = 'none'; }
document.addEventListener('click', e => { if (e.target === document.getElementById('eventModal')) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ===== 图表 =====
function initCharts() {
    drawChart('cpuMemChart', { datasets: [{ label: 'CPU %', data: [], color: '#00b4d8' }, { label: '内存 %', data: [], color: '#aa66ff' }] });
    drawChart('tempChart', { datasets: [{ label: '温度 °C', data: [], color: '#ff4466' }] });
    drawChart('timingChart', { datasets: [{ label: '总计 ms', data: [], color: '#ffaa00' }] });
    drawChart('bandwidthChart', { datasets: [{ label: 'KB/s', data: [], color: '#00b4d8' }] });
}

function updateChartData() {
    if (state.events.length === 0) return;
    const recent = [...state.events].reverse();
    const cpuMem = [], gpu = [], temp = [], timing = [];

    recent.forEach(ev => {
        const sys = parseJsonField(ev.system);
        const t = parseJsonField(ev.timing);
        if (sys) {
            if (sys.cpu_percent != null && sys.memory_percent != null) cpuMem.push({ cpu: sys.cpu_percent, mem: sys.memory_percent });
            if (sys.gpu_load_percent != null) gpu.push(sys.gpu_load_percent);
            if (sys.max_temp_c != null) temp.push(sys.max_temp_c);
        }
        if (t && t.total_ms != null) timing.push(t.total_ms);
    });

    const slice = arr => arr.slice(-MAX_CHART_POINTS);
    const smooth = (arr, a) => emaSmooth(removeSpikes(arr), a || 0.4);

    const cpuSlice = slice(cpuMem);
    drawChart('cpuMemChart', {
        datasets: [
            { label: 'CPU %', data: smooth(cpuSlice.map(p => p.cpu)), color: '#00b4d8' },
            { label: '内存 %', data: smooth(cpuSlice.map(p => p.mem)), color: '#aa66ff' }
        ]
    });
    drawChart('tempChart', { datasets: [{ label: '温度 °C', data: smooth(slice(temp), 0.3), color: '#ff4466' }] });
    drawChart('timingChart', { datasets: [{ label: '总计 ms', data: smooth(slice(timing)), color: '#ffaa00' }] });
}

function removeSpikes(data, threshold) {
    threshold = threshold || 3;
    if (data.length < 3) return data;
    var result = data.slice();
    for (var i = 1; i < data.length - 1; i++) {
        var avg = (result[i - 1] + data[i + 1]) / 2;
        if (Math.abs(data[i] - avg) > Math.abs(data[i + 1] - result[i - 1]) * threshold) result[i] = avg;
    }
    return result;
}

function emaSmooth(data, alpha) {
    alpha = alpha || 0.3;
    if (data.length === 0) return [];
    var result = [data[0]];
    for (var i = 1; i < data.length; i++) result.push(alpha * data[i] + (1 - alpha) * result[i - 1]);
    return result;
}

function drawChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = { top: 20, right: 20, bottom: 30, left: 50 };
    ctx.clearRect(0, 0, W, H);
    const cw = W - pad.left - pad.right, ch = H - pad.top - pad.bottom;

    let maxVal = 0;
    config.datasets.forEach(ds => ds.data.forEach(v => { if (v > maxVal) maxVal = v; }));
    if (maxVal === 0 && config.datasets[0].data.length === 0) {
        ctx.fillStyle = '#556677'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('等待数据...', W / 2, H / 2); return;
    }
    maxVal = Math.max(maxVal * 1.2, 10);

    ctx.strokeStyle = 'rgba(42,58,78,0.5)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + ch * i / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
        ctx.fillStyle = '#556677'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
        ctx.fillText((maxVal - maxVal * i / 4).toFixed(0), pad.left - 5, y + 3);
    }

    config.datasets.forEach(ds => {
        if (ds.data.length === 0) return;
        const pts = ds.data.length, stepX = cw / Math.max(pts - 1, 1);
        ctx.strokeStyle = ds.color; ctx.lineWidth = 2; ctx.beginPath();
        ds.data.forEach((v, i) => {
            const x = pad.left + i * stepX, y = pad.top + ch - (v / maxVal * ch);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = ds.color;
        ds.data.forEach((v, i) => {
            const x = pad.left + i * stepX, y = pad.top + ch - (v / maxVal * ch);
            ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
        });
    });

    let lx = pad.left; const ly = H - 5;
    ctx.font = '10px sans-serif';
    config.datasets.forEach(ds => {
        ctx.fillStyle = ds.color; ctx.fillRect(lx, ly - 8, 12, 8);
        ctx.fillStyle = '#8899aa'; ctx.textAlign = 'left'; ctx.fillText(ds.label, lx + 15, ly);
        lx += ctx.measureText(ds.label).width + 30;
    });
}

// ===== 同步文件 =====
async function refreshSyncedFiles() {
    try {
        const resp = await fetch('/api/files/list');
        const data = await resp.json();
        if (data.ok) renderSyncedFiles(data.files);
    } catch (e) {}
}

function renderSyncedFiles(files) {
    const container = document.getElementById('syncedFilesList');
    if (!files || files.length === 0) {
        container.innerHTML = '<p class="empty-hint">暂无同步文件，检测完成后 CSV 文件会自动同步到此</p>';
        return;
    }
    container.innerHTML = files.map(f => {
        const sizeKb = (f.size / 1024).toFixed(1);
        const typeIcon = f.filename.includes('metrics') ? '📈' : '📊';
        const dateStr = f.date.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3');
        return `<div class="synced-file-item">
            <span class="sf-icon">${typeIcon}</span>
            <span class="sf-name">${escapeHtml(f.filename)}</span>
            <span class="sf-meta">${dateStr}</span>
            <span class="sf-size">${sizeKb} KB</span>
            <a class="sf-btn" href="/api/files/download/${encodeURIComponent(f.path)}" download>下载</a>
        </div>`;
    }).join('');
}

// 启动时加载 + 定时刷新
document.addEventListener('DOMContentLoaded', () => {
    refreshSyncedFiles();
    setInterval(refreshSyncedFiles, 10000);
});

function openOutputDir() {
    showToast('info', '文件位置', 'CSV 文件自动保存在 output/ 目录下，按 session 名归档');
}

// ===== 操作函数 =====
function clearFilters() {
    document.getElementById('eventTypeFilter').value = '';
    state.currentFilter = '';
    renderEventTable();
    showToast('info', '筛选已清空', '');
}

async function exportCSV() {
    const p = state.currentFilter ? `?event_type=${state.currentFilter}` : '';
    window.open(`/api/export/csv${p}`, '_blank');
    showToast('success', '导出成功', 'CSV 已开始下载');
}

async function sendTestEvent() {
    try {
        const resp = await fetch('/api/test/send-sample', { method: 'POST' });
        const data = await resp.json();
        if (data.ok) { showToast('success', '测试事件已发送', `类型: ${data.event_type}`); refreshEvents(); }
        else showToast('error', '发送失败', data.error || '未知错误');
    } catch (e) { showToast('error', '发送失败', e.message); }
}

async function deleteTestEvents() {
    if (!confirm('确定删除所有测试事件？')) return;
    try {
        const resp = await fetch('/api/events/test', { method: 'DELETE' });
        const data = await resp.json();
        if (data.ok) { showToast('success', '删除成功', `已删除 ${data.deleted} 条`); refreshEvents(); refreshStats(); }
    } catch (e) { showToast('error', '删除失败', e.message); }
}

async function deleteAllEvents() {
    if (!confirm('⚠️ 确定删除所有事件？此操作不可恢复！')) return;
    try {
        const resp = await fetch('/api/events/all', { method: 'DELETE' });
        const data = await resp.json();
        if (data.ok) { showToast('success', '删除成功', `已删除 ${data.deleted} 条`); refreshEvents(); refreshStats(); }
    } catch (e) { showToast('error', '删除失败', e.message); }
}

// ===== 工具函数 =====
function parseJsonField(f) { if (!f) return null; if (typeof f === 'object') return f; try { return JSON.parse(f); } catch { return null; } }
function formatTime(s, short) {
    if (!s) return '-';
    try {
        const d = new Date(s);
        return short ? d.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
                     : d.toLocaleString('zh-CN', { hour12: false });
    } catch { return s; }
}
function escapeHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function showToast(type, title, msg) {
    const c = document.getElementById('toastContainer');
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
    t.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span><div class="toast-content"><div class="toast-title">${title}</div>${msg ? `<div class="toast-message">${msg}</div>` : ''}</div>`;
    c.appendChild(t);
    setTimeout(() => { t.style.animation = 'slideOut 0.3s ease'; setTimeout(() => t.remove(), 300); }, 3000);
}

// ===== 远程控制 =====
let lastSendTime = 0;
const THROTTLE_MS = 1000;

async function sendCommand(action) {
    const now = Date.now();
    if (now - lastSendTime < THROTTLE_MS) { showRcToast('请稍候', 'info'); return; }
    lastSendTime = now;
    try {
        const resp = await fetch('/api/commands', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }) });
        const data = await resp.json();
        if (resp.ok) { showRcToast(`已发送: ${formatAction(action)}`, 'success'); setTimeout(refreshHistory, 1000); }
        else showRcToast(`失败: ${data.error}`, 'error');
    } catch (e) { showRcToast(`网络错误: ${e.message}`, 'error'); }
}

async function sendParams() {
    const params = {
        confidence: parseFloat(document.getElementById('rcConfidence').value),
        iou: parseFloat(document.getElementById('rcIou').value),
        frame_interval: parseInt(document.getElementById('rcFrameInterval').value)
    };
    try {
        const resp = await fetch('/api/commands', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'set_param', params }) });
        if (resp.ok) { showRcToast('参数已发送', 'success'); setTimeout(refreshHistory, 1000); }
    } catch (e) { showRcToast(`发送失败: ${e.message}`, 'error'); }
}

async function refreshHistory() {
    try {
        const resp = await fetch('/api/commands/history?limit=10');
        const data = await resp.json();
        renderHistory(data.history);
    } catch (e) {}
}

function renderHistory(history) {
    const c = document.getElementById('rcHistoryList');
    if (!history || !history.length) { c.innerHTML = '<p class="rc-empty">暂无命令记录</p>'; return; }
    c.innerHTML = history.map(cmd => {
        const icon = { pending: '⏳', picked: '🔄', done: '✅', failed: '❌', timeout: '⏱️' }[cmd.status] || '❓';
        const time = new Date(cmd.created_at).toLocaleTimeString('zh-CN');
        const msg = cmd.result ? escapeHtml(cmd.result.message) : '';
        return `<div class="rc-history-item rc-status-${cmd.status}"><span>${icon}</span><span class="rc-hist-action">${escapeHtml(formatAction(cmd.action))}</span><span class="rc-hist-time">${time}</span>${msg ? `<span class="rc-hist-msg">${msg}</span>` : ''}</div>`;
    }).join('');
}

async function refreshDeviceStatus() {
    try {
        const resp = await fetch('/api/devices/status');
        const data = await resp.json();
        const device = data['ELF2-TBJU-01'];
        if (device) {
            const diff = (new Date() - new Date(device.last_seen)) / 1000;
            document.getElementById('ovDeviceId').textContent = device.ip || 'ELF2-TBJU-01';
        }
    } catch (e) {}
}

function formatAction(a) {
    return { open_camera: '打开摄像头', start_detection: '开始检测', stop_detection: '停止检测', pause_detection: '暂停检测', start_capacity: '开始评估', stop_capacity: '停止评估', toggle_recording: '录制', stop_recording: '停止录制', show_status: '系统状态', mute_alarm: '静音报警', unmute_alarm: '解除静音', set_param: '设置参数' }[a] || a;
}

async function refreshLogs() {
    try {
        const resp = await fetch('/api/events/recent?event_type=app_log&limit=30');
        const data = await resp.json();
        renderLogs(data.events || []);
    } catch (e) {}
}

function renderLogs(logs) {
    const c = document.getElementById('rcLogList');
    if (!logs || !logs.length) { c.innerHTML = '<p class="rc-empty">暂无日志</p>'; return; }
    c.innerHTML = logs.map(log => {
        const time = new Date(log.created_at).toLocaleTimeString('zh-CN');
        return `<div class="rc-log-item"><span class="rc-log-time">${escapeHtml(time)}</span><span class="rc-log-msg">${escapeHtml(log.message)}</span></div>`;
    }).join('');
}

function showRcToast(msg, type = 'info') {
    const t = document.createElement('div');
    t.className = `rc-toast rc-toast-${type}`;
    t.style.cssText = 'position:fixed;bottom:20px;right:20px;padding:12px 20px;border-radius:8px;font-size:13px;color:#fff;opacity:0;transform:translateY(10px);transition:all .3s;z-index:9999;box-shadow:0 4px 12px rgba(0,0,0,.3);';
    if (type === 'success') t.style.background = '#2e7d32';
    else if (type === 'error') t.style.background = '#c62828';
    else t.style.background = '#1565c0';
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => { t.style.opacity = '1'; t.style.transform = 'translateY(0)'; }, 10);
    setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateY(10px)'; setTimeout(() => t.remove(), 300); }, 2500);
}

// 参数滑块
document.getElementById('rcConfidence')?.addEventListener('input', function () { document.getElementById('confValue').textContent = this.value; });
document.getElementById('rcIou')?.addEventListener('input', function () { document.getElementById('iouValue').textContent = this.value; });
document.getElementById('rcFrameInterval')?.addEventListener('input', function () { document.getElementById('frameIntValue').textContent = this.value; });
