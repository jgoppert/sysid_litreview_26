"""Fetch provisional or archived benchmark dataset payloads."""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from .registry import load_manifest, source_url


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "work" / "data"


def _filename_from_url(url: str, default: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    return name or default


def _google_drive_file_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    query_id = urllib.parse.parse_qs(parsed.query).get("id")
    if query_id:
        return query_id[0]
    match = re.search(r"/file/d/([^/]+)", parsed.path)
    return match.group(1) if match else None


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "sysid-benchmark-dataset-fetch/0.1"})
    with urllib.request.urlopen(request) as response, output.open("wb") as stream:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--url", default=None, help="override manifest source URL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.dataset_id)
    url = args.url or source_url(manifest)
    if not url:
        raise SystemExit(f"{args.dataset_id} has no source URL")

    temporary_source = manifest.get("temporary_source") or {}
    provider = str(temporary_source.get("provider") or "")
    if provider == "purdue_sharepoint" and args.url is None:
        raise SystemExit(
            f"{args.dataset_id} uses a Purdue SharePoint folder, not a direct archive URL. "
            "Download it manually to work/data/<dataset_id>/raw or pass --url with a direct .zip archive link."
        )

    parsed = urllib.parse.urlparse(url)
    if "drive.google.com" in parsed.netloc and not _google_drive_file_id(url):
        raise SystemExit(
            "Google Drive folder URLs cannot be downloaded reliably without the Drive API. "
            "Use a direct file share URL or pass --url for a single archive."
        )

    download_url = url
    if "drive.google.com" in parsed.netloc:
        file_id = _google_drive_file_id(url)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    output_dir = args.output_dir or DATA_ROOT / args.dataset_id / "downloads"
    output = output_dir / _filename_from_url(url, f"{args.dataset_id}.download")
    print(f"Downloading {args.dataset_id}")
    print(f"  from: {url}")
    print(f"  to:   {output}")
    try:
        _download(download_url, output)
    except Exception as exc:
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
