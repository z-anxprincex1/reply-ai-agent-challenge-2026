from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import run_cli


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_cli(str(root / "dataset" / "deus-ex"), "deus-ex", str(root / "outputs" / "deus-ex.txt"))


if __name__ == "__main__":
    main()
