#!/usr/bin/env python3
from pathlib import Path
import shutil

def main() -> None:
    root = Path(__file__).resolve().parent.parent

    src_dir = root / "tests"
    dst_twig_dir = root / "templates"
    dst_php_dir = root / "templates" / "elements"

    if not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    dst_twig_dir.mkdir(parents=True, exist_ok=True)
    dst_php_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("sketch-*.svg"))
    if not files:
        print(f"No files found matching {src_dir / 'sketch-*.svg'}")
        return

    copied = 0
    for src in files:
        name = src.name  # e.g. sketch-A.svg
        if not (name.startswith("sketch-") and name.endswith(".svg")):
            continue

        letter = name[len("sketch-") : -len(".svg")]  # everything between
        if not letter:
            continue

        dst_twig = dst_twig_dir / f"mmxx-{letter}.svg.twig"
        dst_php = dst_php_dir / f"mmxx-{letter}.php"

        # Copy bytes exactly (SVG content stays unchanged)
        shutil.copyfile(src, dst_twig)
        shutil.copyfile(src, dst_php)

        copied += 1
        print(f"{src} -> {dst_twig}")
        print(f"{src} -> {dst_php}")

    print(f"\nDone. Processed {copied} file(s).")

if __name__ == "__main__":
    main()
