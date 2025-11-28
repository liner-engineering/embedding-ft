"""
Common evaluation utilities for BEIR-based evaluation scripts.
Provides shared functions for running evaluations and saving results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import (
    DenseRetrievalExactSearch as DRES,
)

from embedding.common.base_model import BaseEmbeddingModel


def run_beir_evaluation(
    model: BaseEmbeddingModel,
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    batch_size: int = 256,
    score_function: str = "cos_sim",
    k_values: Optional[List[int]] = None,
) -> tuple:
    """
    Run BEIR evaluation using DenseRetrievalFaissSearch.

    Args:
        model: Embedding model instance
        corpus: Dict[doc_id, Dict[title, text]]
        queries: Dict[query_id, query_text]
        qrels: Dict[query_id, Dict[doc_id, relevance_score]]
        batch_size: Batch size for encoding
        score_function: Scoring function for FAISS ('cos_sim' or 'dot')
        use_faiss_gpu: Whether to use GPU for FAISS index
        k_values: List of k values for evaluation metrics

    Returns:
        Tuple of (results, ndcg, map, recall, precision)
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 50, 100]

    if not queries:
        print("\nNo queries found. Cannot evaluate.")
        return None, None, None, None, None

    # Initialize BEIR's DenseRetrievalExactSearch
    print(f"\nInitializing DenseRetrievalExactSearch with {score_function}...")
    exact_search = DRES(
        model,
        batch_size=batch_size,
    )

    # Initialize retrieval evaluator
    retriever = EvaluateRetrieval(exact_search, score_function=score_function, k_values=k_values)

    # Run evaluation
    print("\nRunning evaluation...")
    results = retriever.retrieve(corpus, queries)

    # Evaluate results (only if qrels exist)
    if qrels:
        print("\nComputing metrics...")
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        return results, ndcg, _map, recall, precision
    print("\nWARNING: No relevance judgments (qrels) found.")
    return results, None, None, None, None


def print_evaluation_results(
    ndcg: Optional[Dict] = None,
    _map: Optional[Dict] = None,
    recall: Optional[Dict] = None,
    precision: Optional[Dict] = None,
    title: str = "EVALUATION RESULTS",
):
    """
    Print evaluation metrics in a formatted way.

    Args:
        ndcg: NDCG scores
        _map: MAP scores
        recall: Recall scores
        precision: Precision scores
        title: Title for the results section
    """
    if not any([ndcg, _map, recall, precision]):
        print("\nNo metrics to display.")
        return

    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    if ndcg:
        print("\nNDCG:")
        for k, v in ndcg.items():
            print(f"  {k}: {v:.4f}")

    if _map:
        print("\nMAP:")
        for k, v in _map.items():
            print(f"  {k}: {v:.4f}")

    if recall:
        print("\nRecall:")
        for k, v in recall.items():
            print(f"  {k}: {v:.4f}")

    if precision:
        print("\nPrecision:")
        for k, v in precision.items():
            print(f"  {k}: {v:.4f}")

    print(f"{'=' * 80}")


def save_evaluation_results(
    output_dir: str,
    output_filename: str,
    results_dict: Dict,
):
    """
    Save evaluation results to a JSON file.

    Args:
        output_dir: Directory to save results
        output_filename: Name of the output file
        results_dict: Dictionary containing all results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / output_filename
    with result_file.open("w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {result_file}")


def build_results_dict(
    metrics: Dict,
    config: Dict,
    metadata: Dict,
) -> Dict:
    """
    Build a comprehensive results dictionary.

    Args:
        metrics: Dictionary with ndcg, map, recall, precision
        config: Configuration parameters used for evaluation
        metadata: Additional metadata (model_name, dataset info, etc.)

    Returns:
        Complete results dictionary ready for saving
    """
    results_dict = {**metadata}

    if metrics.get("ndcg") is not None:
        results_dict["metrics"] = {
            "ndcg": {k: float(v) for k, v in metrics["ndcg"].items()},
            "map": {k: float(v) for k, v in metrics["map"].items()},
            "recall": {k: float(v) for k, v in metrics["recall"].items()},
            "precision": {k: float(v) for k, v in metrics["precision"].items()},
        }

    results_dict["config"] = config

    return results_dict
