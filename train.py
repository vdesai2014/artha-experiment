#!/usr/bin/env python3
"""
ACT Training Script for Artha Platform

This script:
1. Downloads dataset from Artha (if not already present)
2. Trains ACT policy on synthetic data
3. Uploads checkpoints to Artha periodically and on completion

Usage:
    # Full pipeline (download, train, upload)
    python train.py \
        --username testuser \
        --dataset synthetic-v1 \
        --experiment exp-001 \
        --steps 1000

    # Local only (skip download/upload)
    python train.py \
        --data ./local_data \
        --output ./checkpoints \
        --steps 100 \
        --no-upload

Environment:
    ARTHA_API_KEY: API key for Artha platform
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Our custom dataloader
from simple_dataloader import SimpleLeRobotDataset, SimpleDatasetMetadata, collate_fn

# LeRobot imports
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Artha client
from artha import download_dataset
from artha.upload import upload_single_file, upload_checkpoint


def dataset_features_to_policy_features(features: dict) -> dict[str, PolicyFeature]:
    """Convert dataset features to PolicyFeature format."""
    policy_features = {}

    for key, feat_info in features.items():
        shape = tuple(feat_info["shape"])

        if key == "action":
            ft_type = FeatureType.ACTION
        elif key.startswith("observation.images"):
            ft_type = FeatureType.VISUAL
        elif key == "observation.state":
            ft_type = FeatureType.STATE
        elif key == "observation.environment_state":
            ft_type = FeatureType.ENV
        else:
            continue

        policy_features[key] = PolicyFeature(type=ft_type, shape=shape)

    return policy_features


def save_checkpoint(
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    step: int,
    output_dir: Path,
    is_best: bool = False,
) -> Path:
    """Save checkpoint locally."""
    if is_best:
        ckpt_dir = output_dir / "best"
    else:
        ckpt_dir = output_dir / f"step_{step:06d}"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)

    # Save metadata
    meta = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "is_best": is_best,
    }
    import json
    with open(ckpt_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return ckpt_dir


def main():
    parser = argparse.ArgumentParser(description="Train ACT on Artha dataset")

    # Data source (either remote or local)
    parser.add_argument("--username", "-u", help="Artha username (for remote dataset)")
    parser.add_argument("--dataset", "-d", help="Dataset session name (for remote)")
    parser.add_argument("--data", help="Local data directory (skips download)")

    # Experiment config
    parser.add_argument("--experiment", "-e", help="Experiment name for uploads")
    parser.add_argument("--output", "-o", default="./checkpoints", help="Local output directory")

    # Training config
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action chunk size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Checkpoint config
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--upload-every", type=int, default=500, help="Upload checkpoint every N steps")
    parser.add_argument("--no-upload", action="store_true", help="Skip checkpoint uploads")

    # API config
    parser.add_argument("--api-key", help="Artha API key (or set ARTHA_API_KEY)")
    parser.add_argument("--base-url", default="https://api.artha.bot", help="API base URL")

    args = parser.parse_args()

    # Validate arguments
    if args.data:
        data_dir = Path(args.data)
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            sys.exit(1)
    elif args.username and args.dataset:
        data_dir = Path(f"./data/{args.username}/{args.dataset}")
    else:
        print("Error: Provide either --data or both --username and --dataset")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get API key if needed
    api_key = args.api_key or os.environ.get("ARTHA_API_KEY")
    do_upload = not args.no_upload and api_key and args.experiment and args.username

    if not args.no_upload and not do_upload:
        print("Warning: Uploads disabled (missing api_key, experiment, or username)")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("ACT Training - Artha Platform")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Upload enabled: {do_upload}")
    if do_upload:
        print(f"Experiment: {args.experiment}")
    print()

    # Download dataset if needed
    if not data_dir.exists() and args.username and args.dataset:
        print("Downloading dataset...")
        if not api_key:
            print("Error: API key required for download. Set ARTHA_API_KEY.")
            sys.exit(1)

        success = download_dataset(
            username=args.username,
            session_name=args.dataset,
            output_dir=data_dir,
            api_key=api_key,
            base_url=args.base_url,
        )
        if not success:
            print("Error: Dataset download failed!")
            sys.exit(1)
        print()

    # Load metadata
    print("Loading dataset metadata...")
    meta = SimpleDatasetMetadata(data_dir)
    print(f"  FPS: {meta.fps}")
    print(f"  Features: {list(meta.features.keys())}")

    # Convert features
    features = dataset_features_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    print(f"  Input features: {list(input_features.keys())}")
    print(f"  Output features: {list(output_features.keys())}")
    print()

    # Create ACT config and policy
    print("Creating ACT policy...")
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        device=str(device),
    )

    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {num_params:,}")
    print()

    # Create preprocessor
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

    # Create dataset and dataloader
    print("Creating dataloader...")
    dataset = SimpleLeRobotDataset(data_dir, chunk_size=args.chunk_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batches per epoch: {len(dataloader)}")
    print()

    # Optimizer
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # Training loop
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    step = 0
    total_loss = 0.0
    best_loss = float("inf")
    epoch = 0

    while step < args.steps:
        epoch += 1

        for batch in dataloader:
            # Preprocess
            batch = preprocessor(batch)

            # Forward
            loss, loss_dict = policy.forward(batch)

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            step += 1

            # Logging
            if step % 10 == 0 or step == 1:
                avg_loss = total_loss / step
                print(f"Step {step:5d}/{args.steps} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

            # Save checkpoint
            if step % args.save_every == 0:
                print(f"\nSaving checkpoint at step {step}...")
                ckpt_dir = save_checkpoint(
                    policy, preprocessor, postprocessor,
                    step, output_dir, is_best=False
                )
                print(f"  Saved to: {ckpt_dir}")

                # Track best
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_dir = save_checkpoint(
                        policy, preprocessor, postprocessor,
                        step, output_dir, is_best=True
                    )
                    print(f"  New best! Saved to: {best_dir}")

                # Upload if enabled
                if do_upload and step % args.upload_every == 0:
                    print(f"  Uploading checkpoint...")
                    try:
                        upload_checkpoint(
                            checkpoint_dir=ckpt_dir,
                            username=args.username,
                            experiment_name=args.experiment,
                            api_key=api_key,
                            base_url=args.base_url,
                            quiet=True,
                        )
                        print(f"  Uploaded!")
                    except Exception as e:
                        print(f"  Upload failed: {e}")

                print()

            if step >= args.steps:
                break

    # Final checkpoint
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    final_loss = total_loss / step
    print(f"Final avg loss: {final_loss:.4f}")

    print("\nSaving final checkpoint...")
    final_dir = save_checkpoint(
        policy, preprocessor, postprocessor,
        step, output_dir, is_best=False
    )
    print(f"Saved to: {final_dir}")

    # Upload final checkpoint
    if do_upload:
        print("\nUploading final checkpoint...")
        try:
            upload_checkpoint(
                checkpoint_dir=final_dir,
                username=args.username,
                experiment_name=args.experiment,
                api_key=api_key,
                base_url=args.base_url,
            )
            print("Upload complete!")
        except Exception as e:
            print(f"Upload failed: {e}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
