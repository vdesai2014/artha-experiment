"""
Checkpoint Upload Module

Uploads experiment checkpoints to Artha storage via presigned URLs.
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx
from tqdm import tqdm

from .client import ArthaClient


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def upload_file(
    file_path: Path,
    presigned_url: str,
    timeout: float = 300.0,
) -> tuple[str, bool, Optional[str]]:
    """
    Upload a single file to presigned URL.

    Returns:
        (path, success, error_message)
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()

        with httpx.Client(timeout=timeout) as client:
            resp = client.put(presigned_url, content=content)
            resp.raise_for_status()

        return (str(file_path), True, None)

    except Exception as e:
        return (str(file_path), False, str(e))


def upload_checkpoint(
    checkpoint_dir: Path,
    username: str,
    experiment_name: str,
    api_key: Optional[str] = None,
    base_url: str = "https://api.artha.bot",
    asset_type: str = "checkpoint",
    max_workers: int = 5,
    quiet: bool = False,
) -> bool:
    """
    Upload checkpoint files to Artha.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        username: Username for the experiment
        experiment_name: Name of the experiment
        api_key: API key (defaults to ARTHA_API_KEY env var)
        base_url: API base URL
        asset_type: Asset type ("checkpoint", "config", "artifact")
        max_workers: Parallel upload workers
        quiet: Suppress progress output

    Returns:
        True if all files uploaded successfully
    """
    api_key = api_key or os.environ.get("ARTHA_API_KEY")
    if not api_key:
        raise ValueError("API key required. Set ARTHA_API_KEY or pass api_key.")

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find all files to upload
    files_to_upload = list(checkpoint_dir.rglob("*"))
    files_to_upload = [f for f in files_to_upload if f.is_file()]

    if not files_to_upload:
        print("No files found to upload!")
        return False

    if not quiet:
        print(f"Found {len(files_to_upload)} files to upload")
        print("Computing hashes...")

    # Compute hashes for all files
    files_metadata = {}
    for file_path in tqdm(files_to_upload, disable=quiet, desc="Hashing"):
        rel_path = file_path.relative_to(checkpoint_dir)
        file_hash = compute_sha256(file_path)
        files_metadata[str(rel_path)] = {"hash": file_hash}

    if not quiet:
        print(f"Requesting upload URLs for {experiment_name}...")

    # Get presigned URLs
    client = ArthaClient(api_key=api_key, base_url=base_url)
    response = client.request_experiment_upload(
        username=username,
        experiment_name=experiment_name,
        asset_type=asset_type,
        files=files_metadata,
    )
    client.close()

    upload_manifest = response.get("upload_manifest", {})

    if not upload_manifest:
        print("No upload URLs received!")
        return False

    if not quiet:
        print(f"Uploading {len(upload_manifest)} files...")

    # Upload in parallel
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for filename, presigned_url in upload_manifest.items():
            file_path = checkpoint_dir / filename
            future = executor.submit(upload_file, file_path, presigned_url)
            futures[future] = filename

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            disable=quiet,
            desc="Uploading",
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
        print(f"\nComplete: {success_count}/{len(files_to_upload)} files uploaded")
        if fail_count > 0:
            print(f"Failed: {fail_count} files")

    return fail_count == 0


def upload_single_file(
    file_path: Path,
    username: str,
    experiment_name: str,
    api_key: Optional[str] = None,
    base_url: str = "https://api.artha.bot",
    asset_type: str = "checkpoint",
    quiet: bool = False,
) -> bool:
    """
    Upload a single checkpoint file.

    Convenience function for uploading individual checkpoints during training.

    Args:
        file_path: Path to the checkpoint file
        username: Username for the experiment
        experiment_name: Name of the experiment
        api_key: API key
        base_url: API base URL
        asset_type: Asset type
        quiet: Suppress output

    Returns:
        True if upload successful
    """
    api_key = api_key or os.environ.get("ARTHA_API_KEY")
    if not api_key:
        raise ValueError("API key required.")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Compute hash
    file_hash = compute_sha256(file_path)
    filename = file_path.name

    if not quiet:
        print(f"Uploading {filename} to {experiment_name}...")

    # Get presigned URL
    client = ArthaClient(api_key=api_key, base_url=base_url)
    response = client.request_experiment_upload(
        username=username,
        experiment_name=experiment_name,
        asset_type=asset_type,
        files={filename: {"hash": file_hash}},
    )
    client.close()

    presigned_url = response.get("upload_manifest", {}).get(filename)
    if not presigned_url:
        print("Failed to get upload URL!")
        return False

    # Upload
    _, success, error = upload_file(file_path, presigned_url)

    if success:
        if not quiet:
            print(f"Uploaded: {filename}")
        return True
    else:
        print(f"Upload failed: {error}")
        return False


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Upload checkpoint to Artha")
    parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint file or directory")
    parser.add_argument("--username", "-u", required=True, help="Username")
    parser.add_argument("--experiment", "-e", required=True, help="Experiment name")
    parser.add_argument("--api-key", help="API key (or set ARTHA_API_KEY)")
    parser.add_argument("--base-url", default="https://api.artha.bot", help="API base URL")
    parser.add_argument("--asset-type", default="checkpoint", choices=["checkpoint", "config", "artifact"])
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.is_dir():
        success = upload_checkpoint(
            checkpoint_dir=checkpoint_path,
            username=args.username,
            experiment_name=args.experiment,
            api_key=args.api_key,
            base_url=args.base_url,
            asset_type=args.asset_type,
            max_workers=args.workers,
            quiet=args.quiet,
        )
    else:
        success = upload_single_file(
            file_path=checkpoint_path,
            username=args.username,
            experiment_name=args.experiment,
            api_key=args.api_key,
            base_url=args.base_url,
            asset_type=args.asset_type,
            quiet=args.quiet,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
