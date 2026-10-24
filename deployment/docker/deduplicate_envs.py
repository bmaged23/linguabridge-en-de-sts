"""Hardlink byte-identical libraries across immutable image environments."""
import hashlib
import os
from pathlib import Path
import stat


def main():
    buckets = {}
    saved = 0
    for folder in ('.venv/lib', '.venv-asr/lib', '.venv-tts-gpu/lib'):
        for path in Path(folder).rglob('*'):
            if path.is_symlink() or not path.is_file():
                continue
            info = path.stat()
            if info.st_size < 4096:
                continue
            key = (info.st_size, stat.S_IMODE(info.st_mode))
            candidates = buckets.setdefault(key, [])
            if not candidates:
                candidates.append((path, None))
                continue
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').digest()
            duplicate = False
            for index, (old, old_digest) in enumerate(candidates):
                if old_digest is None:
                    with old.open('rb') as stream:
                        old_digest = hashlib.file_digest(stream, 'sha256').digest()
                    candidates[index] = (old, old_digest)
                if old_digest == digest:
                    path.unlink()
                    os.link(old, path)
                    saved += info.st_size
                    duplicate = True
                    break
            if not duplicate:
                candidates.append((path, digest))
    print(f'Identical environment files hardlinked: {saved / 1024**2:.1f} MiB saved')


if __name__ == '__main__':
    main()
