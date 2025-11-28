# Academic Embedding Model Evaluation

This directory contains utilities for benchmarking HuggingFace embedding models on academic-focused MTEB tasks and specialized scientific retrieval benchmarks.

## Overview

### Available Scripts

| Script | Description |
|--------|-------------|
| `evaluate_academic_mteb.py` | Evaluate on MTEB academic retrieval tasks |
| `evaluate_qasa.py` | Evaluate on QASA (Question Answering on Scientific Articles) |
| `evaluate_litsearch.py` | Evaluate on LitSearch (Scientific Literature Search) |

## Evaluation Datasets

### MTEB Academic Tasks (Retrieval Only)

- **SciFact**: Retrieve supporting evidence for scientific claims
- **TREC-COVID**: Rank COVID-19 related documents
- **NFCorpus**: Retrieve nutrition-centric documents
- **SCIDOCS**: Predict citations between scientific papers

### Specialized Scientific Retrieval Benchmarks

#### QASA (Question Answering on Scientific Articles)
- **Description**: Retrieval benchmark for question answering on scientific papers
- **Format**: Parquet corpus + JSONL queries
- You can download this dataset from [QASA evaluation dataset](https://huggingface.co/datasets/LinerAI/QASA)

#### LitSearch
- **Description**: Retrieval benchmark for scientific literature search with 597 realistic queries about ML/NLP papers
- **Format**: HuggingFace datasets (`princeton-nlp/LitSearch`)
- **Reference**: [EMNLP 2024 Paper](https://arxiv.org/abs/2407.18940)

---

## MTEB Academic Evaluation

### Basic Usage

```bash
python -m embedding.evaluate.evaluate_academic_mteb \
    --model_name "sentence-transformers/all-MiniLM-L6-v2" \
    --output_dir "results/minilm"
```

### Custom Pooling

Four pooling strategies are available: `cls`, `avg`, `last`, `weightedavg`

```bash
# Mean pooling
python -m embedding.evaluate.evaluate_academic_mteb \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --pool_type avg \
    --output_dir "results/qwen-avg"

# CLS pooling
python -m embedding.evaluate.evaluate_academic_mteb \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --pool_type cls \
    --output_dir "results/qwen-cls"

# Last token pooling (for LLM-style embeddings)
python -m embedding.evaluate.evaluate_academic_mteb \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --pool_type last \
    --output_dir "results/qwen-last"
```

### With Encoding Method

```bash
python -m embedding.evaluate.evaluate_academic_mteb \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --pool_type avg \
    --encoding_method query_or_passage \
    --output_dir "results/qwen"
```

### Matryoshka Representation Learning (MRL)

Evaluate with truncated embedding dimensions:

```bash
python -m embedding.evaluate.evaluate_academic_mteb \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --matryoshka_dim 256 \
    --output_dir "results/qwen-mrl-256"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | (required) | HuggingFace model name or path |
| `--output_dir` | `results` | Directory to save results |
| `--tasks` | all | Specific tasks: `SciFact`, `TRECCOVID`, `NFCorpus`, `SCIDOCS` |
| `--batch_size` | `128` | Batch size for encoding |
| `--pool_type` | `None` | Pooling strategy: `cls`, `avg`, `last`, `weightedavg` |
| `--encoding_method` | `None` | Encoding method (see below) |
| `--normalize` | `True` | L2 normalize embeddings |
| `--max_length` | `4096` | Maximum sequence length |
| `--general_instruction` | (default) | Instruction for instruction-based models |
| `--matryoshka_dim` | `None` | Dimension to truncate embeddings to |

---

## QASA Evaluation

Evaluate models on the QASA benchmark for Question Answering on Scientific Articles.

### Basic Usage

```bash
python -m embedding.evaluate.evaluate_qasa \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --corpus_path "/path/to/qasa_section.parquet" \
    --query_path "/path/to/qasa_data_qasa_test.jsonl" \
    --output_dir "results/qwen"
```

### With Custom Settings

```bash
python -m embedding.evaluate.evaluate_qasa \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --corpus_path "/path/to/qasa_section.parquet" \
    --query_path "/path/to/qasa_data_qasa_test.jsonl" \
    --pool_type avg \
    --encoding_method query_or_passage \
    --batch_size 64 \
    --output_dir "results/qwen"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | (required) | HuggingFace model name or path |
| `--corpus_path` | `qasa_section.parquet` | Path to QASA corpus parquet file |
| `--query_path` | `qasa_data_qasa_test.jsonl` | Path to QASA query JSONL file |
| `--output_dir` | `results` | Directory to save results |
| `--batch_size` | `128` | Batch size for encoding |
| `--pool_type` | `None` | Pooling strategy: `cls`, `avg`, `last`, `weightedavg` |
| `--encoding_method` | `None` | Encoding method (see below) |
| `--normalize` | `True` | L2 normalize embeddings |
| `--max_length` | `4096` | Maximum sequence length |
| `--score_function` | `cos_sim` | Scoring function: `cos_sim`, `dot` |
| `--general_instruction` | (default) | Instruction for instruction-based models |
| `--matryoshka_dim` | `None` | Dimension to truncate embeddings to |

---

## LitSearch Evaluation

Evaluate models on the LitSearch benchmark for scientific literature search.

### Basic Usage

```bash
python -m embedding.evaluate.evaluate_litsearch \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --output_dir "results/qwen"
```

### With Custom Settings

```bash
python -m embedding.evaluate.evaluate_litsearch \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --corpus_key "title_abstract" \
    --pool_type avg \
    --encoding_method query_or_passage \
    --batch_size 64 \
    --output_dir "results/qwen"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | (required) | HuggingFace model name or path |
| `--dataset_name` | `princeton-nlp/LitSearch` | HuggingFace dataset name |
| `--corpus_config` | `corpus_clean` | Corpus configuration |
| `--query_config` | `query` | Query configuration |
| `--corpus_split` | `full` | Split name for corpus |
| `--query_split` | `full` | Split name for queries |
| `--corpus_key` | `title_abstract` | Document text field: `title`, `abstract`, `title_abstract`, `full_paper` |
| `--output_dir` | `results` | Directory to save results |
| `--batch_size` | `128` | Batch size for encoding |
| `--pool_type` | `None` | Pooling strategy: `cls`, `avg`, `last`, `weightedavg` |
| `--encoding_method` | `None` | Encoding method (see below) |
| `--normalize` | `True` | L2 normalize embeddings |
| `--max_length` | `4096` | Maximum sequence length |
| `--score_function` | `cos_sim` | Scoring function: `cos_sim`, `dot` |
| `--general_instruction` | (default) | Instruction for instruction-based models |
| `--matryoshka_dim` | `None` | Dimension to truncate embeddings to |

### Dataset Info

LitSearch automatically downloads from HuggingFace Datasets:
- **Corpus**: ~360K scientific papers from arXiv
- **Queries**: 597 realistic literature search queries
- **Relevance judgments**: Ground truth relevance annotations

---

## Encoding Methods

The `--encoding_method` parameter controls how queries and passages are formatted:

| Method | Description |
|--------|-------------|
| `no-prefix` | No prefix added |
| `query_or_passage` | Add "query: " or "passage: " prefix (E5-style) |
| `query` | Add "query: " prefix to queries only |
| `instruction` | Use task-specific instructions |
| `general_instruction` | Use `--general_instruction` for all queries |
| `chat_user_assistant` | Chat template with user/assistant roles |
| `chat_query_passage` | Chat template with query/passage format |
| `embedding_gemma` | Embedding-Gemma specific format |

---

## Output Format

```
Task: SciFact
----------------------------------------
  NDCG@10: 0.6789
  MAP@10:  0.6234
  Recall@10: 0.8912
```
