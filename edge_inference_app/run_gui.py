#!/usr/bin/env python3
"""启动 GUI 应用 — PyQt5 优先，tkinter 后备"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 修复 OpenCV 自带 Qt 插件与 PyQt5 的冲突
# OpenCV 捆绑了 Qt 插件，会导致 PyQt5 加载失败
# 必须在 import cv2 或 PyQt5 之前清除这个路径
if sys.platform == 'linux':
    cv2_qt_plugin = Path('/usr/local/lib/python3.10/dist-packages/cv2/qt/plugins')
    if cv2_qt_plugin.exists():
        # 不让 OpenCV 的 Qt 插件干扰 PyQt5
        os.environ.pop('QT_PLUGIN_PATH', None)
        # 设置为系统 Qt 插件路径
        for p in ['/usr/lib/aarch64-linux-gnu/qt5/plugins',
                  '/usr/lib/x86_64-linux-gnu/qt5/plugins',
                  '/usr/lib/qt5/plugins']:
            if Path(p).exists():
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = p
                break

try:
    import PyQt5
    from src.gui.tbju_demo_gui import main
    print('使用 PyQt5 GUI')
    main()
except ImportError:
    print('PyQt5 不可用，使用 tkinter GUI')
    from src.gui.tbju_demo_tk import main
    main()
