#!/usr/bin/env python3
from pathlib import Path
import shutil

def main() -> None:
    root = Path(__file__).resolve().parent.parent  # tools/ -> project root
    src_dir = root / "tests"
    dst_dir = root / "src"

    if not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("sketch-*.svg"))
    if not files:
        print(f"No files found matching: {src_dir / 'sketch-*.svg'}")
        return

    copied = 0
    for src in files:
        name = src.name  # sketch-X.svg
        if not (name.startswith("sketch-") and name.endswith(".svg")):
            continue

        letter = name[len("sketch-") : -len(".svg")]
        if not letter:
            continue

        dst = dst_dir / f"character-{letter}.svg"
        shutil.copy2(src, dst)
        copied += 1
        print(f"{src} -> {dst}")

    print(f"\nDone. Copied {copied} file(s) to {dst_dir}.")

if __name__ == "__main__":
    main()
