"""Resumable HF dataset download with completeness verification.

Strategy:
- List remote file set via API (with 429 backoff).
- Filter to the allowlist (skip unused camera views).
- For each missing file, hf_hub_download with backoff.
- ThreadPoolExecutor(max_workers=3) — keeps below the 5000 req / 5min auth cap.
- Verify all files exist on disk; raise if any are missing.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from threading import Lock

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    raise RuntimeError("Set HF_TOKEN env var (https://huggingface.co/settings/tokens)")
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/OXE_LEROBOT")
NUM_WORKERS = 3

DATASETS = [
    {
        "repo_id": "IPEC-COMMUNITY/bridge_orig_lerobot",
        "local_dir": f"{DATA_ROOT}/bridge_orig_1.0.0_lerobot",
        "allow_patterns": [
            "*.json",
            "*.md",
            ".gitattributes",
            "meta/*",
            "data/**/*.parquet",
            "videos/**/observation.images.image_0/*.mp4",
        ],
    },
    {
        "repo_id": "IPEC-COMMUNITY/fractal20220817_data_lerobot",
        "local_dir": f"{DATA_ROOT}/fractal20220817_data_0.1.0_lerobot",
        "allow_patterns": [
            "*.json",
            "*.md",
            ".gitattributes",
            "meta/*",
            "data/**/*.parquet",
            "videos/**/observation.images.image/*.mp4",
        ],
    },
]

_lock = Lock()
_done_count = 0
_total = 0
_last_log = time.time()


def matches_any(path, patterns):
    return any(fnmatch(path, pat) for pat in patterns)


def with_backoff(fn, *args, retries=20, base_delay=30, max_delay=600, **kwargs):
    delay = base_delay
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                with _lock:
                    print(f"[{time.strftime('%H:%M:%S')}] 429, sleeping {delay}s (retry {i+1}/{retries})", flush=True)
                time.sleep(delay)
                delay = min(int(delay * 1.5), max_delay)
            else:
                raise
        except Exception as e:
            # Treat transient network errors as retryable too.
            msg = str(e).lower()
            if any(s in msg for s in ("timed out", "connection", "temporary", "incomplete read", "remotedisconnected")):
                with _lock:
                    print(f"[{time.strftime('%H:%M:%S')}] transient {type(e).__name__}, sleeping {delay}s", flush=True)
                time.sleep(delay)
                delay = min(int(delay * 1.5), max_delay)
            else:
                raise
    raise RuntimeError(f"exhausted retries for {fn.__name__}: {args} {kwargs}")


def list_remote_files(api, repo_id):
    return with_backoff(api.list_repo_files, repo_id=repo_id, repo_type="dataset", token=TOKEN)


def download_one(repo_id, fname, local_dir):
    with_backoff(
        hf_hub_download,
        repo_id=repo_id,
        filename=fname,
        repo_type="dataset",
        local_dir=local_dir,
        token=TOKEN,
    )
    global _done_count, _last_log
    with _lock:
        _done_count += 1
        if time.time() - _last_log > 30 or _done_count == _total:
            pct = 100.0 * _done_count / _total
            print(f"[{time.strftime('%H:%M:%S')}] {_done_count}/{_total} ({pct:.1f}%) downloaded", flush=True)
            _last_log = time.time()


def download_dataset(spec):
    global _done_count, _total, _last_log
    api = HfApi()
    print(f"\n[{time.strftime('%H:%M:%S')}] LISTING {spec['repo_id']}", flush=True)
    all_files = list_remote_files(api, spec["repo_id"])
    wanted = [f for f in all_files if matches_any(f, spec["allow_patterns"])]
    print(f"  remote total: {len(all_files)}  wanted: {len(wanted)}", flush=True)

    local = Path(spec["local_dir"])
    local.mkdir(parents=True, exist_ok=True)
    missing = [f for f in wanted if not (local / f).exists()]
    print(f"  already on disk: {len(wanted) - len(missing)}  missing: {len(missing)}", flush=True)

    if not missing:
        print(f"[{time.strftime('%H:%M:%S')}] DONE {spec['repo_id']} (nothing to do)", flush=True)
        return

    _done_count = 0
    _total = len(missing)
    _last_log = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [pool.submit(download_one, spec["repo_id"], f, str(local)) for f in missing]
        for fut in as_completed(futures):
            fut.result()  # propagate any error after retries

    still_missing = [f for f in wanted if not (local / f).exists()]
    if still_missing:
        raise RuntimeError(f"{spec['repo_id']}: {len(still_missing)} files still missing after download")
    print(f"[{time.strftime('%H:%M:%S')}] DONE {spec['repo_id']} ({len(wanted)} files)", flush=True)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for spec in DATASETS:
        if only and only not in spec["repo_id"]:
            continue
        download_dataset(spec)
    print("\nAll datasets verified complete.")
