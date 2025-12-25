"""
Dataset Download Module

Downloads datasets from Artha storage via the data.artha.bot worker.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx
from tqdm import tqdm

from .client import ArthaClient


def flatten_manifest(manifest: dict, prefix: str = "") -> dict[str, dict]:
    """
    Flatten nested manifest to {path: {url, size_bytes}}.

    The manifest from API is hierarchical, we need flat paths for downloading.
    """
    flat = {}

    for key, value in manifest.items():
        current_path = f"{prefix}/{key}" if prefix else key

        if isinstance(value, dict):
            if "url" in value:
                # Leaf node with URL
                flat[current_path] = value
            else:
                # Nested directory
                flat.update(flatten_manifest(value, current_path))
        else:
            # Direct URL string (shouldn't happen but handle it)
            flat[current_path] = {"url": value, "size_bytes": None}

    return flat


def download_file(
    url: str,
    output_path: Path,
    api_key: str,
    timeout: float = 60.0,
) -> tuple[str, bool, Optional[str]]:
    """
    Download a single file.

    Returns:
        (path, success, error_message)
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers={"X-API-Key": api_key})
            resp.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(resp.content)

        return (str(output_path), True, None)

    except Exception as e:
        return (str(output_path), False, str(e))


def download_dataset(
    username: str,
    session_name: str,
    output_dir: Path,
    api_key: Optional[str] = None,
    base_url: str = "https://api.artha.bot",
    max_workers: int = 10,
    quiet: bool = False,
) -> bool:
    """
    Download a complete dataset.

    Args:
        username: Dataset owner
        session_name: Dataset session name
        output_dir: Local output directory
        api_key: API key (defaults to ARTHA_API_KEY env var)
        base_url: API base URL
        max_workers: Parallel download workers
        quiet: Suppress progress output

    Returns:
        True if all files downloaded successfully
    """
    api_key = api_key or os.environ.get("ARTHA_API_KEY")
    if not api_key:
        raise ValueError("API key required. Set ARTHA_API_KEY or pass api_key.")

    output_dir = Path(output_dir)

    # Get manifest
    if not quiet:
        print(f"Fetching manifest for {username}/{session_name}...")

    client = ArthaClient(api_key=api_key, base_url=base_url)
    manifest_response = client.get_dataset_manifest(username, session_name)
    client.close()

    download_manifest = manifest_response.get("download_manifest", {})
    files = flatten_manifest(download_manifest)

    if not files:
        print("No files in manifest!")
        return False

    total_files = len(files)
    total_size = sum(f.get("size_bytes", 0) or 0 for f in files.values())

    if not quiet:
        print(f"Downloading {total_files} files ({total_size / 1024 / 1024:.2f} MB)...")

    # Download in parallel
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for file_path, file_info in files.items():
            output_path = output_dir / file_path
            future = executor.submit(
                download_file,
                file_info["url"],
                output_path,
                api_key,
            )
            futures[future] = file_path

        # Progress bar
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            disable=quiet,
            desc="Downloading",
        )

        for future in pbar:
            path, success, error = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                if not quiet:
                    tqdm.write(f"Failed: {path} - {error}")

    if not quiet:
        print(f"\nComplete: {success_count}/{total_files} files downloaded")
        if fail_count > 0:
            print(f"Failed: {fail_count} files")

    return fail_count == 0


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Download dataset from Artha")
    parser.add_argument("--username", "-u", required=True, help="Dataset owner username")
    parser.add_argument("--session", "-s", required=True, help="Dataset session name")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--api-key", help="API key (or set ARTHA_API_KEY)")
    parser.add_argument("--base-url", default="https://api.artha.bot", help="API base URL")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    success = download_dataset(
        username=args.username,
        session_name=args.session,
        output_dir=Path(args.output),
        api_key=args.api_key,
        base_url=args.base_url,
        max_workers=args.workers,
        quiet=args.quiet,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
