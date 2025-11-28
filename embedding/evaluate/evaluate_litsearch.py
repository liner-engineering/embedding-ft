"""
LitSearch Evaluation Script using BEIR
Evaluates embedding models on LitSearch dataset for scientific literature search.

LitSearch is a retrieval benchmark for scientific literature search consisting of
597 realistic queries about recent ML and NLP papers.

This script:
1. Loads LitSearch data from HuggingFace datasets
2. Prepares corpus and queries in BEIR format
3. Uses BEIR's DenseRetrievalExactSearch for evaluation
4. Computes standard retrieval metrics (NDCG, MAP, Recall, etc.)

Reference: https://github.com/princeton-nlp/LitSearch
Paper: LitSearch: A Retrieval Benchmark for Scientific Literature Search (EMNLP 2024)
"""

import json
from typing import Dict, Optional, Tuple

from datasets import load_dataset

from embedding.common.base_model import BaseEmbeddingModel
from embedding.evaluate.common_evaluation import (
    build_results_dict,
    print_evaluation_results,
    run_beir_evaluation,
    save_evaluation_results,
)


def prepare_beir_data(
    corpus_data,
    query_data,
    corpus_key: str = "title_abstract",
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Prepare LitSearch data in BEIR format.

    Args:
        corpus_data: HuggingFace dataset with corpus documents
        query_data: HuggingFace dataset with queries and relevance judgments
        corpus_key: Which field to use as document text ('title', 'abstract', 'title_abstract', 'full_paper')

    Returns:
        Tuple of (corpus, queries, qrels) in BEIR format
        - corpus: Dict[doc_id, Dict[title, text]]
        - queries: Dict[query_id, query_text]
        - qrels: Dict[query_id, Dict[doc_id, relevance_score]]
    """
    # Prepare corpus
    corpus = {}
    corpusid_to_doc_id = {}  # Map from corpusid to doc_id

    for idx, item in enumerate(corpus_data):
        doc_id = f"doc_{idx}"
        corpusid = item.get("corpusid", "")  # int or str
        title = str(item.get("title", ""))
        abstract = str(item.get("abstract", ""))

        # Choose text based on corpus_key
        if corpus_key == "title":
            text = title
        elif corpus_key == "abstract":
            text = abstract
        elif corpus_key == "title_abstract":
            text = f"{title}\n\n{abstract}"
        elif corpus_key == "full_paper" and "full_paper" in item:
            text = str(item.get("full_paper", ""))
        else:
            # Default to title + abstract
            text = f"{title}\n\n{abstract}"

        corpus[doc_id] = {
            "title": title,
            "text": text,
        }

        # Map corpusid to doc_id for matching with queries
        if corpusid:
            corpusid_to_doc_id[str(corpusid)] = doc_id

    print(f"Created corpus with {len(corpus)} documents")
    print(f"Sample corpus keys: {list(corpus.keys())[:3]}")

    # Prepare queries and qrels
    queries = {}
    qrels = {}

    for query_idx, query_item in enumerate(query_data):
        query_id = f"query_{query_idx}"
        query_text = str(query_item.get("query", ""))
        queries[query_id] = query_text

        # Prepare qrels (ground truth relevance)
        # LitSearch uses 'corpusids' field (list of relevant corpus IDs)
        corpusids = query_item.get("corpusids", [])

        if corpusids:
            qrels[query_id] = {}

            for corpusid in corpusids:
                # Convert to string for matching
                corpusid_str = str(corpusid)

                # Find matching doc_id
                if corpusid_str in corpusid_to_doc_id:
                    doc_id = corpusid_to_doc_id[corpusid_str]
                    qrels[query_id][doc_id] = 1  # Binary relevance

    print(f"Created {len(queries)} queries")
    print(f"Created qrels for {len(qrels)} queries")

    return corpus, queries, qrels


def evaluate_litsearch(
    model_name: str,
    dataset_name: str = "princeton-nlp/LitSearch",
    corpus_config: str = "corpus_clean",
    query_config: str = "query",
    query_split: str = "full",
    corpus_split: str = "full",
    corpus_key: str = "title_abstract",
    output_dir: str = "results",
    pool_type: Optional[str] = None,
    encoding_method: Optional[str] = None,
    normalize: bool = True,
    max_length: int = 4096,
    batch_size: int = 128,
    score_function: str = "cos_sim",
    general_instruction: str = "Given a scientific query, retrieve relevant research papers",
    matryoshka_dim: Optional[int] = None,
):
    """
    Evaluate embedding model on LitSearch dataset using BEIR's DenseRetrievalExactSearch.

    Args:
        model_name: HuggingFace model name or path
        dataset_name: HuggingFace dataset name (default: "princeton-nlp/LitSearch")
        corpus_config: Dataset configuration for corpus (default: "corpus_clean")
        query_config: Dataset configuration for queries (default: "query")
        query_split: Split name for queries (default: "full")
        corpus_split: Split name for corpus (default: "full")
        corpus_key: Which field to use as document text ('title', 'abstract', 'title_abstract', 'full_paper')
        output_dir: Directory to save results
        pool_type: Custom pooling strategy (None=auto, or 'cls'/'avg'/'last'/'weightedavg')
        encoding_method: Custom encoding method (None=auto)
        normalize: Whether to L2 normalize embeddings
        max_length: Maximum sequence length
        batch_size: Batch size for encoding
        score_function: Scoring function for Exact Search ('cos_sim' or 'dot')
        general_instruction: Instruction for instruction-based models
    """
    print(f"Loading LitSearch dataset from {dataset_name}...")
    print(f"Corpus config: {corpus_config}, Query config: {query_config}")

    # Load corpus and queries from HuggingFace
    corpus_data = load_dataset(dataset_name, corpus_config, split=corpus_split)
    query_data = load_dataset(dataset_name, query_config, split=query_split)

    # Initialize model
    print(f"\nLoading model: {model_name}")
    model = BaseEmbeddingModel(
        model_name=model_name,
        pool_type=pool_type,
        encoding_method=encoding_method,
        normalize=normalize,
        max_length=max_length,
        batch_size=batch_size,
        general_instruction=general_instruction,
        matryoshka_dim=matryoshka_dim,
    )

    # Prepare data in BEIR format
    print("\nPreparing data in BEIR format...")
    corpus, queries, qrels = prepare_beir_data(
        corpus_data=corpus_data,
        query_data=query_data,
        corpus_key=corpus_key,
    )

    # Run evaluation
    results, ndcg, _map, recall, precision = run_beir_evaluation(
        model=model,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        batch_size=batch_size,
        score_function=score_function,
        k_values=[1, 3, 5, 10, 20, 50, 100],
    )

    if results is None:
        return None

    # Print results
    print_evaluation_results(
        ndcg=ndcg,
        _map=_map,
        recall=recall,
        precision=precision,
        title="EVALUATION RESULTS - LitSearch",
    )

    # Prepare and save results (only if metrics exist)
    if ndcg is not None:
        metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "corpus_config": corpus_config,
            "query_config": query_config,
            "corpus_key": corpus_key,
            "num_queries": len(queries),
            "num_corpus": len(corpus),
            "num_qrels": len(qrels),
        }

        config = {
            "pool_type": pool_type,
            "encoding_method": encoding_method,
            "normalize": normalize,
            "max_length": max_length,
            "batch_size": batch_size,
            "score_function": score_function,
            "matryoshka_dim": matryoshka_dim,
        }

        metrics = {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
        }

        results_dict = build_results_dict(
            metrics=metrics,
            config=config,
            metadata=metadata,
        )

        output_filename = "litsearch_exact_search_results.json"
        if matryoshka_dim:
            output_filename = f"litsearch_exact_search_results_dim{matryoshka_dim}.json"

        save_evaluation_results(
            output_dir=output_dir,
            output_filename=output_filename,
            results_dict=results_dict,
        )

        return results_dict

    print("\nSkipping result save (no metrics available)")
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on LitSearch dataset using BEIR"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/LitSearch",
        help="HuggingFace dataset name (default: 'princeton-nlp/LitSearch')",
    )
    parser.add_argument(
        "--corpus_config",
        type=str,
        default="corpus_clean",
        help="Dataset configuration for corpus (default: 'corpus_clean')",
    )
    parser.add_argument(
        "--query_config",
        type=str,
        default="query",
        help="Dataset configuration for queries (default: 'query')",
    )
    parser.add_argument(
        "--query_split",
        type=str,
        default="full",
        help="Split name for queries (default: 'full')",
    )
    parser.add_argument(
        "--corpus_split",
        type=str,
        default="full",
        help="Split name for corpus (default: 'full')",
    )
    parser.add_argument(
        "--corpus_key",
        type=str,
        default="title_abstract",
        choices=["title", "abstract", "title_abstract", "full_paper"],
        help="Which field to use as document text (default: 'title_abstract')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results (default: 'results')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--pool_type",
        type=str,
        default=None,
        choices=["cls", "avg", "last", "weightedavg"],
        help="Custom pooling strategy (default: None, auto-detect)",
    )
    parser.add_argument(
        "--encoding_method",
        type=str,
        default=None,
        choices=[
            "no-prefix",
            "query_or_passage",
            "query",
            "instruction",
            "general_instruction",
            "chat_user_assistant",
            "chat_query_passage",
            "embedding_gemma",
        ],
        help="Encoding method for queries/passages (default: None, auto-detect)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2 normalize embeddings (default: True)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization (default: 4096)",
    )
    parser.add_argument(
        "--score_function",
        type=str,
        default="cos_sim",
        choices=["cos_sim", "dot"],
        help="Scoring function for Exact Search (default: 'cos_sim')",
    )
    parser.add_argument(
        "--general_instruction",
        type=str,
        default="Given a scientific query, retrieve relevant research papers",
        help="General instruction for instruction-based encoding",
    )
    parser.add_argument(
        "--matryoshka_dim",
        type=int,
        default=None,
        help="Dimension to truncate embeddings to",
    )

    args = parser.parse_args()

    # Print configuration
    print("Configuration:")
    print(json.dumps(vars(args), indent=2))

    # Run evaluation
    evaluate_litsearch(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        corpus_config=args.corpus_config,
        query_config=args.query_config,
        query_split=args.query_split,
        corpus_split=args.corpus_split,
        corpus_key=args.corpus_key,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        pool_type=args.pool_type,
        encoding_method=args.encoding_method,
        normalize=args.normalize,
        max_length=args.max_length,
        score_function=args.score_function,
        general_instruction=args.general_instruction,
        matryoshka_dim=args.matryoshka_dim,
    )


if __name__ == "__main__":
    main()
