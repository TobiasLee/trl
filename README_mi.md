# Reward Model Training & Evaluation Framework

A comprehensive framework for training and evaluating vision-language reward models using various architectures including Qwen2.5-VL, MIMO, and other multimodal models.

## Overview

This framework supports:
- **Multi-scale Models**: 3B, 7B, 32B, 72B parameter models
- **Multiple Architectures**: Qwen2.5-VL, MIMO-VL, and custom models  
- **Robustness Training**: Irrelevant query augmentation
- **Distributed Training**: Multi-GPU support with DeepSpeed
- **Comprehensive Evaluation**: Automated checkpoint evaluation with metrics

## Environment Setup

### Installation

First, install the package in development mode:

```bash
# Clone the repository and install in editable mode
pip install -e .

# Install with development dependencies (optional)
pip install -e .[dev]
```

### Requirements

- Python 3.8+
- CUDA-compatible GPU(s)
- PyTorch with CUDA support
- Transformers, TRL, and other dependencies (installed via pip)

## Quick Start

### Training a Reward Model

```bash
# Single GPU training
python train_rm.py training_configs/7b_config.yaml

# Multi-GPU distributed training (recommended)
./run_distributed.sh --gpus 8 --config training_configs/7b_config.yaml

# With custom arguments
./run_distributed.sh --gpus 4 --port 29501 --config training_configs/qwen3b_config_balance_irrelevant.yaml
```

### Evaluating Trained Models

```bash
# Evaluate with default config
./run_eval.sh

# Evaluate with custom config
./run_eval.sh training_configs/eval_config_example7b.yaml

# Specify GPU
./run_eval.sh training_configs/eval_config_example.yaml 0
```

## Training Configuration

### Available Model Configurations

Located in `training_configs/`:

#### Base Model Configs
- `7b_config.yaml` - Standard 7B model configuration
- `32b_config.yaml` - Large 32B model configuration  
- `72b_config.yaml` - Extra large 72B model configuration

#### Qwen Model Configs
- `qwen3b_config_balance_vanilla.yaml` - Basic 3B Qwen model
- `qwen3b_config_balance_irrelevant.yaml` - 3B with irrelevant query augmentation
- `qwen7b_config_balance_vanilla.yaml` - Basic 7B Qwen model
- `qwen7b_config_balance_irrelevant.yaml` - 7B with robustness training

#### MIMO Model Configs
- `mimo7b_config_balance_vanilla.yaml` - Basic MIMO 7B model
- `mimo7b_config_balance_irrelevant.yaml` - MIMO 7B with augmentation
- `mimo7bRL_config_balance_aug2.yaml` - MIMO 7B for RL training

### Key Configuration Parameters

```yaml
# Model settings
model_name_or_path: /path/to/model
attn_implementation: flash_attention_2  # or "sdpa"
min_pixels: 6272    # 8 * 28 * 28
max_pixels: 1605632 # 4096 * 28 * 28

# Data settings
dataset_path: /path/to/preference/data
max_seq_length: 3072
max_train_samples: null  # Use all data

# Robustness training (optional)
use_irrelevant_queries: true
num_irrelevant_pairs: 1
irrelevant_loss_weight: 1.0
freeze_vit: false  # Whether to freeze vision encoder

# Training hyperparameters
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
warmup_steps: 100

# Optimization
gradient_checkpointing: true
disable_dropout: true
deepspeed: zero2.json
bf16: true
```

## Training Features

### Irrelevant Query Augmentation
Improves model robustness by training with irrelevant image-text pairs:
```yaml
use_irrelevant_queries: true
num_irrelevant_pairs: 1      # Number of irrelevant pairs per sample
irrelevant_loss_weight: 1.0  # Weight for irrelevant loss
```

### Vision Encoder Freezing
Option to freeze the vision transformer during training:
```yaml
freeze_vit: true  # Freeze ViT parameters, only train text components
```

### Multi-GPU Training
Distributed training with `run_distributed.sh`:
```bash
# Options
--gpus N        # Number of GPUs (default: 8)
--port PORT     # Master port (default: 29500)  
--config FILE   # YAML/JSON config file
-- EXTRA_ARGS   # Additional arguments for train_rm.py
```

### DeepSpeed Integration
Supports DeepSpeed ZeRO for large model training:
```yaml
deepspeed: zero2.json  # DeepSpeed configuration
```

## Evaluation

### Evaluation Configuration

Create evaluation configs in `training_configs/`:

```yaml
# Example eval_config_example.yaml
model_dirs:
  - "vlm_reward_model_balance_100_en_zh_total30k"
  - "mimo_vlm_reward_model_balance_100_en_zh_total30k"

dataset_path: "/path/to/evaluation/data"
base_model_path: "/path/to/base/model"

checkpoint_pattern: "checkpoint-*"  # Evaluate all checkpoints
eval_batch_size: 2
max_eval_samples: null  # Use all evaluation data

output_file: "evaluation_results.csv"
```

### Evaluation Metrics

The evaluation produces:
- **Preference Accuracy**: How often model ranks preferred response higher
- **Average Margin**: Mean score difference between preferred/rejected responses  
- **Margin Std**: Standard deviation of score differences
- **Sample Count**: Number of evaluation samples

### Evaluation Output

Results saved in multiple formats:
- `evaluation_results.csv` - Tabular results for all models/checkpoints
- `evaluation_results.json` - Detailed JSON results
- `logs/eval_TIMESTAMP.log` - Full evaluation logs

## Directory Structure

```
├── train_rm.py              # Main training script
├── eval_rm.py               # Evaluation script  
├── run_distributed.sh       # Distributed training wrapper
├── run_eval.sh              # Evaluation wrapper
├── training_configs/        # Model configurations
│   ├── 7b_config.yaml
│   ├── qwen3b_config_balance_irrelevant.yaml
│   ├── mimo7b_config_balance_vanilla.yaml
│   └── eval_config_example.yaml
├── logs/                    # Training and evaluation logs
└── evaluation_results/      # Evaluation outputs
```

## Model Support

### Supported Architectures
- **Qwen2.5-VL**: 3B, 7B, 32B, 72B variants
- **MIMO-VL**: Custom multimodal architecture
- **Custom Models**: Extensible to other vision-language models

### Attention Implementations  
- `flash_attention_2` (recommended for speed)
- `sdpa` (standard scaled dot-product attention)

### Precision Support
- `bf16` (recommended)
- `fp16` 
- `fp32`

## Advanced Usage

### Custom Dataset Loading
The framework expects preference datasets with image-text pairs:
```python
{
    "chosen": "Better response text",
    "rejected": "Worse response text", 
    "images": [PIL.Image objects or paths]
}
```

### Memory Optimization
For large models, use:
- DeepSpeed ZeRO-2 or ZeRO-3
- Gradient checkpointing
- Smaller batch sizes with gradient accumulation
- ViT freezing to reduce trainable parameters

### Monitoring Training
- Logs saved to `logs/` directory
- Integration with experiment tracking (configurable)
- Checkpoint saving at regular intervals

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size, enable gradient checkpointing, use DeepSpeed
2. **Slow training**: Enable flash attention, use bf16, increase batch size
3. **Convergence issues**: Adjust learning rate, warmup steps, or freeze ViT

### Performance Tips

- Use `flash_attention_2` for faster training
- Enable `bf16` on supported hardware
- Tune `gradient_accumulation_steps` for optimal GPU utilization
- Consider freezing ViT for faster convergence on limited data

## Examples

### Training a 3B Model with Robustness
```bash
./run_distributed.sh \
  --gpus 4 \
  --config training_configs/qwen3b_config_balance_irrelevant.yaml
```

### Evaluating Multiple Models
```bash
./run_eval.sh training_configs/eval_config_example7b.yaml 0
```

### Training with Custom Arguments
```bash
./run_distributed.sh \
  --gpus 8 \
  --config training_configs/7b_config.yaml \
  -- --learning_rate 5e-6 --num_train_epochs 5
```
