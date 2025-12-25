"""
Minimal LeRobot-compatible dataloader for v2.1 synthetic datasets.

This dataloader outputs batches in the exact format that LeRobot's ACT policy expects,
allowing us to test our synthetic data with the real LeRobot training pipeline.

Output format per sample:
{
    "observation.state": (state_dim,),
    "observation.images.gripper_left": (C, H, W),
    "observation.images.gripper_right": (C, H, W),
    "observation.images.overhead": (C, H, W),
    "action": (chunk_size, action_dim),
    "action_is_pad": (chunk_size,),
    "timestamp": (1,),
    "frame_index": (1,),
    "episode_index": (1,),
    "index": (1,),
    "task_index": (1,),
}
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SimpleLeRobotDataset(Dataset):
    """
    Minimal dataloader for v2.1 LeRobot datasets.

    Designed to be a drop-in replacement for LeRobotDataset when testing
    with the actual LeRobot ACT training code.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        chunk_size: int = 100,
        image_transforms: transforms.Compose | None = None,
    ):
        """
        Args:
            dataset_path: Path to the v2.1 dataset root directory
            chunk_size: Number of future actions to return (action chunking)
            image_transforms: Optional torchvision transforms for images
        """
        self.root = Path(dataset_path)
        self.chunk_size = chunk_size
        self.image_transforms = image_transforms or transforms.Compose([
            transforms.ToTensor(),  # Converts to (C, H, W) and scales to [0, 1]
        ])

        # Load metadata
        self.info = self._load_json(self.root / "meta" / "info.json")
        self.stats = self._load_json(self.root / "meta" / "stats.json")
        self.episodes = self._load_jsonl(self.root / "meta" / "episodes.jsonl")
        self.tasks = self._load_jsonl(self.root / "meta" / "tasks.jsonl")

        # Load all parquet data into memory (fine for small datasets)
        self.data = self._load_all_parquet()

        # Build episode index mapping
        self._build_episode_index()

        # Extract camera keys from info
        self.camera_keys = [
            k for k in self.info["features"].keys()
            if k.startswith("observation.images.")
        ]

        print(f"Loaded dataset: {len(self.data)} frames, {len(self.episodes)} episodes")
        print(f"Camera keys: {self.camera_keys}")
        print(f"Action chunk size: {self.chunk_size}")

    def _load_json(self, path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    def _load_jsonl(self, path: Path) -> list[dict]:
        items = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def _load_all_parquet(self) -> pd.DataFrame:
        """Load all parquet files into a single DataFrame."""
        parquet_dir = self.root / "data"
        parquet_files = sorted(parquet_dir.glob("**/*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True).sort_values("index").reset_index(drop=True)

    def _build_episode_index(self):
        """Build mapping from episode_index to frame range."""
        self.episode_ranges = {}
        for ep in self.episodes:
            ep_idx = ep["episode_index"]
            mask = self.data["episode_index"] == ep_idx
            indices = self.data[mask].index.tolist()
            if indices:
                self.episode_ranges[ep_idx] = (min(indices), max(indices))

    def __len__(self) -> int:
        return len(self.data)

    @property
    def num_frames(self) -> int:
        """Compatibility with LeRobot."""
        return len(self.data)

    @property
    def num_episodes(self) -> int:
        """Compatibility with LeRobot."""
        return len(self.episodes)

    @property
    def fps(self) -> int:
        """Compatibility with LeRobot."""
        return self.info.get("fps", 30)

    @property
    def features(self) -> dict:
        """Compatibility with LeRobot - return feature definitions."""
        return self.info.get("features", {})

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample with action chunking.

        Returns dict with:
        - observation.state: current state
        - observation.images.*: current camera images
        - action: (chunk_size, action_dim) future actions
        - action_is_pad: (chunk_size,) padding mask
        - metadata fields (timestamp, frame_index, etc.)
        """
        row = self.data.iloc[idx]
        ep_idx = int(row["episode_index"])
        ep_start, ep_end = self.episode_ranges[ep_idx]

        # Build output dict
        item = {}

        # Metadata
        item["timestamp"] = torch.tensor([row["timestamp"]], dtype=torch.float32)
        item["frame_index"] = torch.tensor([row["frame_index"]], dtype=torch.int64)
        item["episode_index"] = torch.tensor([ep_idx], dtype=torch.int64)
        item["index"] = torch.tensor([row["index"]], dtype=torch.int64)
        item["task_index"] = torch.tensor([row["task_index"]], dtype=torch.int64)

        # Current state
        state = row["observation.state"]
        if isinstance(state, np.ndarray):
            item["observation.state"] = torch.tensor(state, dtype=torch.float32)
        else:
            item["observation.state"] = torch.tensor(list(state), dtype=torch.float32)

        # Action chunking: get next chunk_size actions
        actions = []
        is_pad = []

        for delta in range(self.chunk_size):
            future_idx = idx + delta

            if future_idx <= ep_end:
                # Valid future frame within episode
                future_row = self.data.iloc[future_idx]
                action = future_row["action"]
                if isinstance(action, np.ndarray):
                    actions.append(torch.tensor(action, dtype=torch.float32))
                else:
                    actions.append(torch.tensor(list(action), dtype=torch.float32))
                is_pad.append(False)
            else:
                # Pad with last valid action
                last_row = self.data.iloc[ep_end]
                action = last_row["action"]
                if isinstance(action, np.ndarray):
                    actions.append(torch.tensor(action, dtype=torch.float32))
                else:
                    actions.append(torch.tensor(list(action), dtype=torch.float32))
                is_pad.append(True)

        item["action"] = torch.stack(actions)  # (chunk_size, action_dim)
        item["action_is_pad"] = torch.tensor(is_pad, dtype=torch.bool)  # (chunk_size,)

        # Load images
        for cam_key in self.camera_keys:
            img = self._load_image(ep_idx, int(row["frame_index"]), cam_key)
            item[cam_key] = img

        return item

    def _load_image(self, episode_idx: int, frame_idx: int, cam_key: str) -> torch.Tensor:
        """Load a single image and apply transforms."""
        # Extract camera name from key (e.g., "observation.images.overhead" -> "overhead")
        cam_name = cam_key.replace("observation.images.", "")

        # Build image path for v2.1 format
        img_path = (
            self.root / "images" / "chunk-000" / cam_name /
            f"episode_{episode_idx:06d}" / f"frame_{frame_idx:06d}.png"
        )

        if not img_path.exists():
            # Fallback: try alternative naming
            img_path = (
                self.root / "images" / "chunk-000" / cam_name /
                f"episode_{episode_idx:06d}" / f"frame_{frame_idx:05d}.png"
            )

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        return self.image_transforms(img)


class SimpleDatasetMetadata:
    """
    Minimal metadata class compatible with LeRobot's LeRobotDatasetMetadata.

    Provides the interface needed by ACTConfig and make_pre_post_processors.
    """

    def __init__(self, dataset_path: str | Path):
        self.root = Path(dataset_path)

        # Load metadata files
        self.info = self._load_json(self.root / "meta" / "info.json")
        self._stats = self._load_json(self.root / "meta" / "stats.json")
        self.episodes = self._load_jsonl(self.root / "meta" / "episodes.jsonl")
        self.tasks = self._load_jsonl(self.root / "meta" / "tasks.jsonl")

    def _load_json(self, path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    def _load_jsonl(self, path: Path) -> list[dict]:
        items = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    @property
    def features(self) -> dict:
        """Return feature definitions from info.json."""
        return self.info.get("features", {})

    @property
    def stats(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return dataset statistics as tensors (for normalization)."""
        stats_tensors = {}
        for key, stat_dict in self._stats.items():
            stats_tensors[key] = {
                stat_name: torch.tensor(values, dtype=torch.float32)
                for stat_name, values in stat_dict.items()
            }
        return stats_tensors

    @property
    def fps(self) -> int:
        return self.info.get("fps", 30)

    @property
    def camera_keys(self) -> list[str]:
        return [k for k in self.features.keys() if k.startswith("observation.images.")]

    @property
    def video_keys(self) -> list[str]:
        """For compatibility - we use images not videos."""
        return []


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Custom collate function that stacks tensors properly.

    Handles the case where some values are already tensors and some might be lists.
    """
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values

    return collated


# Quick test
if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "./test_dataset"

    print(f"Testing dataloader with dataset: {dataset_path}")
    print("=" * 60)

    # Test metadata
    meta = SimpleDatasetMetadata(dataset_path)
    print(f"FPS: {meta.fps}")
    print(f"Features: {list(meta.features.keys())}")
    print(f"Camera keys: {meta.camera_keys}")
    print(f"Stats keys: {list(meta.stats.keys())}")

    print("\n" + "=" * 60)

    # Test dataset
    dataset = SimpleLeRobotDataset(dataset_path, chunk_size=10)

    print(f"\nDataset length: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")

    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {type(val)}")

    # Test dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))

    print(f"\nBatch keys: {list(batch.keys())}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    print("\n" + "=" * 60)
    print("Dataloader test passed!")
