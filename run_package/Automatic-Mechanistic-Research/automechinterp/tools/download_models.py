"""Check / download the HF models this project uses (config.py's TARGET_MODEL_ID,
HEAVY_MODEL_ID, SMOKETEST_MODEL_ID).

Usage:
    python -m automechinterp.tools.download_models --check
    python -m automechinterp.tools.download_models --download gpt2 Qwen/Qwen2.5-7B-Instruct
    python -m automechinterp.tools.download_models --check --download-missing
"""
from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

from automechinterp import config

DEFAULT_MODELS = [
    config.TARGET_MODEL_ID,
    config.HEAVY_MODEL_ID,
    config.SMOKETEST_MODEL_ID,
]


def is_downloaded(model_id: str) -> bool:
    """True iff every file in the model's latest snapshot is already cached
    locally -- checked with local_files_only so it never touches the network."""
    try:
        snapshot_download(model_id, local_files_only=True)
        return True
    except LocalEntryNotFoundError:
        return False


def download(model_id: str) -> None:
    print(f"downloading {model_id} ...")
    path = snapshot_download(model_id, resume_download=True)
    print(f"  -> cached at {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report cache status for the default models")
    ap.add_argument("--download", nargs="*", metavar="MODEL_ID", help="download these model ids")
    ap.add_argument("--download-missing", action="store_true",
                     help="with --check, also download any model reported missing")
    args = ap.parse_args()

    if args.download:
        for model_id in args.download:
            download(model_id)
        return 0

    if args.check or args.download_missing:
        missing = []
        for model_id in DEFAULT_MODELS:
            status = "OK" if is_downloaded(model_id) else "MISSING"
            print(f"{status:7s} {model_id}")
            if status == "MISSING":
                missing.append(model_id)
        if args.download_missing:
            for model_id in missing:
                download(model_id)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
