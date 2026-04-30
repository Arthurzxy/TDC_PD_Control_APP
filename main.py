from __future__ import annotations

import sys
from pathlib import Path

# 兼容两种启动方式：
# 1. python -m app.main
# 2. python app/main.py
# 直接运行脚本时，sys.path[0] 会落在 app/ 目录下，此时 `import app...`
# 会失败，所以这里把工程根目录补进 sys.path。
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from PyQt5 import QtGui, QtWidgets

from app.control import AppController
from app.ui.main_window import MainWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("TDC Host V1")
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 11))
    window = MainWindow()
    controller = AppController(window)
    window.controller = controller
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
