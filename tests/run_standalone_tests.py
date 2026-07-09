from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    test_path = Path(__file__).with_name("test_bot_core.py")
    spec = importlib.util.spec_from_file_location("test_bot_core", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {test_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    passed = 0
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        getattr(module, name)()
        print(f"PASS {name}")
        passed += 1
    print(f"{passed} tests passed")


if __name__ == "__main__":
    main()
