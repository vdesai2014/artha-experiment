"""
Artha API Client

Handles authentication and communication with api.artha.bot
"""

import os
from pathlib import Path
from typing import Optional
import httpx


class ArthaClient:
    """Client for Artha API (api.artha.bot)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.artha.bot",
        timeout: float = 30.0,
    ):
        """
        Initialize Artha client.

        Args:
            api_key: API key (defaults to ARTHA_API_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("ARTHA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set ARTHA_API_KEY env var or pass api_key argument."
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers={"X-API-Key": self.api_key},
        )

    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            resp = self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    def get_dataset_manifest(self, username: str, session_name: str) -> dict:
        """
        Get download manifest for a dataset.

        Args:
            username: Dataset owner username
            session_name: Dataset session name

        Returns:
            Manifest dict with download URLs and file info
        """
        resp = self._client.get(
            f"{self.base_url}/manifest/dataset/{username}/{session_name}"
        )
        resp.raise_for_status()
        return resp.json()

    def request_experiment_upload(
        self,
        username: str,
        experiment_name: str,
        asset_type: str,
        files: dict[str, dict],
    ) -> dict:
        """
        Request presigned URLs for experiment upload.

        Args:
            username: User uploading the experiment
            experiment_name: Name of the experiment
            asset_type: Type of asset ("checkpoint", "config", "artifact")
            files: Dict of filename -> metadata (must include "hash")

        Returns:
            Upload manifest with presigned URLs
        """
        resp = self._client.post(
            f"{self.base_url}/upload/experiment",
            json={
                "username": username,
                "experiment_name": experiment_name,
                "asset_type": asset_type,
                "files": files,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
