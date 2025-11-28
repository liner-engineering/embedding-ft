# Embedding Model Training & Evaluation Framework

A comprehensive framework for training and evaluating text embedding models, with specialized support for academic and scientific document retrieval.

## Features

- **Contrastive Learning**: Train embedding models using hard negative sampling
- **Multiple Pooling Strategies**: Support for CLS, mean, last token, and weighted average pooling
- **MRL (Matryoshka Representation Learning)**: Train models with multiple embedding dimensions
- **Academic Benchmarks**: Evaluate on MTEB academic tasks, QASA, and LitSearch

## Project Structure

```
embedding/
├── common/          # Shared utilities and base classes
│   ├── base_model.py    # Base embedding model class
│   ├── config.py        # Model configurations
│   ├── heads.py         # Model head implementations
│   └── utils.py         # Utility functions
├── train/           # Training scripts and configurations
│   ├── src/             # Training source code
│   ├── config/          # Training configurations
│   └── script/          # Training shell scripts
└── evaluate/        # Evaluation scripts
    ├── evaluate_academic_mteb.py   # MTEB academic tasks
    ├── evaluate_qasa.py            # QASA benchmark
    └── evaluate_litsearch.py       # LitSearch benchmark
```

## Installation

```bash
# Create and activate conda environment
conda create -n train_embedding python=3.10
conda activate train_embedding

# Install dependencies
conda install -c conda-forge pyarrow -y
pip install -e .
```

## Quick Start

### Training

```bash
# Run training with default configuration
cd embedding/train
bash script/train.sh
```

### Evaluation

```bash
# Evaluate on MTEB academic tasks
python -m embedding.evaluate.evaluate_academic_mteb \
    --model_name "your-model-path" \
    --output_dir "results/"

# Evaluate on LitSearch
python -m embedding.evaluate.evaluate_litsearch \
    --model_name "your-model-path" \
    --output_dir "results/"

# Evaluate on QASA
python -m embedding.evaluate.evaluate_qasa \
    --model_name "your-model-path" \
    --corpus_path "/path/to/qasa_section.parquet" \
    --query_path "/path/to/qasa_data_qasa_test.jsonl" \
    --output_dir "results/"
```

## Documentation

- [Training](embedding/train/README.md) - Training configuration and scripts
- [Evaluation](embedding/evaluate/README.md) - Benchmark evaluation

## Supported Models

The framework supports various embedding model architectures:

- **E5 Models**: `intfloat/e5-*`
- **Qwen3 Embedding**: `Qwen/Qwen3-Embedding-*`
- **Snowflake Arctic**: `Snowflake/snowflake-arctic-embed-*`
- **Custom Models**: Any HuggingFace transformer model

## License

Apache License 2.0
