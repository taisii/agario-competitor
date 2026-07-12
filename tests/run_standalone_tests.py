from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path


def main() -> None:
    passed = 0
    skipped = 0
    failures: list[tuple[str, BaseException]] = []
    for test_path in sorted(Path(__file__).parent.glob("test_*.py")):
        module_name = test_path.stem
        spec = importlib.util.spec_from_file_location(module_name, test_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {test_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            qualified_name = f"{module_name}.{name}"
            test = getattr(module, name)
            required_parameters = [
                parameter.name
                for parameter in inspect.signature(test).parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if required_parameters:
                print(
                    f"SKIP {qualified_name} "
                    f"(requires fixtures: {', '.join(required_parameters)})"
                )
                skipped += 1
                continue
            try:
                test()
            except BaseException as exc:  # report every independent failure
                print(f"FAIL {qualified_name}: {type(exc).__name__}: {exc}")
                failures.append((qualified_name, exc))
            else:
                print(f"PASS {qualified_name}")
                passed += 1
    print(f"{passed} passed, {len(failures)} failed, {skipped} skipped")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
