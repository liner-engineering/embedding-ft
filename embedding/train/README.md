# Training Module

This module provides scripts and configurations for training embedding models using contrastive learning.

## Quick Start

```bash
# Run training
bash script/train.sh
```

The training script (`script/train.sh`) trains the model and copies necessary configuration files from the source model to the final checkpoint, enabling model loading from the trained checkpoint.

## Configuration

### Dataset Configuration

Dataset configuration files are located in `config/dataset_config/`. Example configuration:

```yaml
use_all: False  # If True, use all data (ignores num_total_data and data_ratio)
num_total_data: 200000
allow_duplicate: True
datasets:
  name: [dataset1, dataset2]
  data_path: [./data/dataset1.json, ./data/dataset2.json]
  data_ratio: [0.7, 0.3]
```

#### Configuration Options

**`use_all: True`**
- Uses all data from the specified datasets
- `num_total_data` and `data_ratio` are ignored

**`use_all: False`**
- Samples `num_total_data` samples from datasets according to `data_ratio`
- If a dataset has fewer samples than allocated:
  - `allow_duplicate: False` - Uses all available data without repetition (total may be less than `num_total_data`)
  - `allow_duplicate: True` - Repeats data to reach the allocated count, with random sampling for the remainder

**Example:**
- `num_total_data: 200000`, `data_ratio: [0.7, 0.3]`
- Dataset1 allocation: 140,000 samples, but only has 60,000 available
- With `allow_duplicate: False`: Uses 60,000 from Dataset1
- With `allow_duplicate: True`: Uses Dataset1 twice (120,000) + randomly samples 20,000 more

### Training Configuration

Training configurations are in `config/train_config/`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `model_name_or_path` | Base model path or HuggingFace model ID |
| `output_dir` | Directory to save checkpoints |
| `max_length` | Maximum sequence length |
| `per_device_train_batch_size` | Batch size per GPU |
| `gradient_accumulation_steps` | Gradient accumulation steps |
| `learning_rate` | Learning rate |
| `num_train_epochs` | Number of training epochs |
| `mrl_dims` | MRL dimensions (e.g., `[768, 512, 256]`) |

## Sanity Test

To quickly verify the training pipeline with minimal data:

```bash
# Enable sanity test mode
python src/train.py --sanity_test
```

When `sanity_test=True`:
- Uses `config/dataset_config/test1.yaml` for data
- Automatically sets small values for batch size, gradient accumulation, and save steps
- Suitable for debugging and verification

## Training Scripts

| Script | Description |
|--------|-------------|
| `script/train.sh` | Main training script |
| `script/train_sanity_test.sh` | Sanity test training |

## Output

After training completes:
1. Model checkpoints are saved to `output_dir`
2. Required configuration files (tokenizer, model config) are copied from the source model
3. The final checkpoint can be loaded directly using `AutoModel.from_pretrained()`
