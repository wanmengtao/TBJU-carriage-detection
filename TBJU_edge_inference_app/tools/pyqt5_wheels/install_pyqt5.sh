#!/bin/bash
# 离线安装 PyQt5 — 在 RK3588 板端执行
# 基于 Ubuntu 22.04 (jammy) + Python 3.10 + aarch64
# 用法: sudo bash install_pyqt5.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== PyQt5 离线安装脚本 ==="
echo ""

# 检查是否 root
if [ "$EUID" -ne 0 ]; then
    echo "请用 sudo 执行: sudo bash install_pyqt5.sh"
    exit 1
fi

# 检查 Python 版本
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python 版本: $PY_VER"

if [ "$PY_VER" != "3.10" ]; then
    echo "错误: .deb 包是为 Python 3.10 编译的，当前版本 $PY_VER 不兼容"
    echo "请确认板端 Python 版本: python3 --version"
    exit 1
fi

# 检查 Qt5 库是否已安装
echo ""
echo "检查 Qt5 依赖..."
QT5_MISSING=""
for lib in libQt5Core5 libQt5Gui5 libQt5Widgets5 libQt5Network5; do
    if ldconfig -p | grep -q "$lib"; then
        echo "  $lib: 已安装"
    else
        echo "  $lib: 未找到"
        QT5_MISSING="$QT5_MISSING $lib"
    fi
done

if [ -n "$QT5_MISSING" ]; then
    echo ""
    echo "缺少 Qt5 运行库:$QT5_MISSING"
    echo "请先安装 Qt5 库（需要临时联网或用 apt-offline）:"
    echo "  sudo apt-get install -y libqt5core5a libqt5gui5 libqt5widgets5 libqt5network5"
    echo ""
    read -p "是否继续尝试安装 PyQt5？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 安装 deb 包（按依赖顺序）
echo ""
echo "安装 Python SIP..."
dpkg -i python3-sip_4.19.25_arm64.deb
# sip-dev 依赖其他包，跳过不影响 PyQt5 运行
dpkg -i python3-sip-dev_4.19.25_arm64.deb 2>/dev/null || echo "  sip-dev 跳过（不影响运行）"

echo "安装 PyQt5 SIP 绑定..."
dpkg -i python3-pyqt5-sip_12.9.1_arm64.deb

echo "安装 PyQt5..."
# 检查是否有被 Breaks 的包
echo "检查可能冲突的包..."
for pkg in calibre python3-pyqt5.qsci python3-pyqt5.qtchart python3-pyqt5.qtwebengine; do
    if dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        echo "  警告: $pkg 已安装，PyQt5 可能要求其更新或卸载"
    fi
done
dpkg -i python3-pyqt5_5.15.6_arm64.deb

# 修复可能的依赖问题
echo ""
echo "检查依赖完整性..."
apt-get install -f -y 2>/dev/null || true

# 验证
echo ""
echo "验证安装..."
if python3 -c "import PyQt5; print('PyQt5 版本:', PyQt5.QtCore.PYQT_VERSION_STR)" 2>/dev/null; then
    echo ""
    echo "=== PyQt5 安装成功! ==="
else
    echo ""
    echo "=== PyQt5 导入失败 ==="
    echo "检查具体错误:"
    python3 -c "import PyQt5" 2>&1
    echo ""
    echo "常见原因:"
    echo "  1. Qt5 运行库缺失 → sudo apt-get install -y libqt5core5a libqt5gui5 libqt5widgets5"
    echo "  2. Python 版本不匹配 → python3 --version 确认是 3.10"
    echo "  3. 架构不匹配 → dpkg --print-architecture 确认是 arm64"
fi
