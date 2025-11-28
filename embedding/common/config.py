"""Configuration constants for embedding models."""

# Default encoding methods for known models
ENCODING_METHOD = {
    "bge-m3": "no-prefix",
    "multilingual-e5-large-instruct": "instruction",
    "multilingual-e5-large": "query_or_passage",
    "multilingual-e5-base": "query_or_passage",
    "multilingual-e5-small": "query_or_passage",
    "snowflake-arctic-embed-l-v2.0": "query",
    "snowflake-arctic-embed-m-v2.0": "query",
    "Qwen3-Embedding-0.6B": "instruction",
    "Qwen3-Embedding-4B": "instruction",
    "Qwen3-Embedding-8B": "instruction",
    "embeddinggemma-300m": "embedding_gemma",
}

# Default pooling methods for known models
POOLING_METHOD = {
    "bge-m3": "avg",  # not used, just placeholder
    "multilingual-e5-large-instruct": "avg",
    "multilingual-e5-large": "avg",
    "multilingual-e5-base": "avg",
    "multilingual-e5-small": "avg",
    "snowflake-arctic-embed-l-v2.0": "cls",
    "snowflake-arctic-embed-m-v2.0": "cls",
    "Qwen3-Embedding-0.6B": "last",
    "Qwen3-Embedding-4B": "last",
    "Qwen3-Embedding-8B": "last",
    "embeddinggemma-300m": "avg",
}
