import os
import sys
import shutil
from pathlib import Path

# Extensions to include in the artifact
INCLUDE_EXTENSIONS = {
    ".exe", ".dll", ".so", ".dylib",
}

# Specific files to always include (e.g. data assets)
INCLUDE_FILES = {
}

def flatten_files(input_dir: str, target_dir: str):
    input_path = Path(input_dir).resolve()
    target_path = Path(target_dir).resolve()

    target_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for root, _, files in os.walk(input_path):
        for filename in files:
            src_file = Path(root) / filename

            # Resolve symlinks to real files
            if src_file.is_symlink():
                real_file = src_file.resolve()
                if not real_file.is_file():
                    continue
                src_file = real_file

            ext = src_file.suffix.lower()
            if ext not in INCLUDE_EXTENSIONS and filename.lower() not in INCLUDE_FILES:
                skipped += 1
                continue

            dst_file = target_path / src_file.name

            try:
                shutil.copy2(src_file, dst_file)
                print(f"  copy: {src_file.relative_to(input_path)}")
                copied += 1
            except Exception as e:
                print(f"  skip {src_file}: {e}")

    print(f"\nDone! Copied {copied} files, skipped {skipped} files.")
    print(f"Output: {target_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage:")
        print("  python flatten_bin_files.py input_dir target_dir")
        print("example:")
        print("  python flatten_bin_files.py build bin")
        sys.exit(1)

    input_dir = sys.argv[1]
    target_dir = sys.argv[2]
    flatten_files(input_dir, target_dir)
