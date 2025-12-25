# Artha ACT Training Experiment

Train ACT (Action Chunking Transformer) on synthetic robot learning data using the Artha platform.

## Quick Start

### Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train on local data
python train.py \
    --data ./path/to/dataset \
    --output ./checkpoints \
    --steps 1000 \
    --no-upload
```

### Remote Training (with Artha)

```bash
# Set your API key
export ARTHA_API_KEY=sk_your_key_here

# Train with dataset download and checkpoint upload
python train.py \
    --username testuser \
    --dataset synthetic-v1 \
    --experiment exp-001 \
    --steps 1000
```

### SkyPilot (Cloud GPU)

```bash
# Launch on cloud GPU
sky launch skypilot/train_act.yaml \
    --env ARTHA_API_KEY=$ARTHA_API_KEY \
    --env ARTHA_USERNAME=testuser \
    --env DATASET_NAME=synthetic-v1 \
    --env EXPERIMENT_NAME=exp-001
```

## Project Structure

```
artha-experiment/
├── train.py              # Main training script
├── simple_dataloader.py  # LeRobot-compatible dataset loader
├── coherent_function.py  # Synthetic data generation logic
├── requirements.txt      # Python dependencies
├── artha/                # Artha client library
│   ├── client.py         # API client
│   ├── download.py       # Dataset download
│   └── upload.py         # Checkpoint upload
└── skypilot/
    └── train_act.yaml    # SkyPilot job config
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--username` | - | Artha username |
| `--dataset` | - | Dataset session name |
| `--experiment` | - | Experiment name for uploads |
| `--data` | - | Local data path (skips download) |
| `--output` | `./checkpoints` | Output directory |
| `--steps` | 1000 | Training steps |
| `--batch-size` | 8 | Batch size |
| `--chunk-size` | 50 | Action chunk size |
| `--save-every` | 500 | Save checkpoint interval |
| `--upload-every` | 500 | Upload checkpoint interval |
| `--no-upload` | False | Skip uploads |

## Data Format

Expected LeRobot v2.1 format:

```
dataset/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/chunk-000/
│   └── episode_*.parquet
└── images/chunk-000/
    ├── gripper_left/
    ├── gripper_right/
    └── overhead/
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ARTHA_API_KEY` | API key for Artha platform |

## License

MIT
