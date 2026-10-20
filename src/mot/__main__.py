"""``python -m mot`` 入口：转发到接口层 CLI。"""

from __future__ import annotations

import sys

from mot.interfaces.cli import main

if __name__ == "__main__":
    sys.exit(main())
