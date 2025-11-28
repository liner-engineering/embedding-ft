"""
Academic Embedding Evaluation Script
Evaluates embedding models on academic-focused MTEB tasks using the latest MTEB API.

Tasks:
- SciFact (Retrieval)
- TREC-COVID (Retrieval)
- NFCorpus (Retrieval)
- SCIDOCS (Retrieval)

Requirements:
    pip install mteb transformers torch
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import mteb
from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

# from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer

from embedding.common.base_model import BaseEmbeddingModel

# Academic-focused MTEB tasks (Retrieval only)
ACADEMIC_TASKS = [
    "SciFact",
    "TRECCOVID",
    "NFCorpus",
    "SCIDOCS",
]


# TODO: update mteb version (2.1.1) and then remove these functions (https://github.com/embeddings-benchmark/mteb/issues/3478)
def _clip_to_uint16(score) -> int:
    try:
        value = int(float(score))
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _load_qrels_with_clipping(self):  # type: ignore[override]
    config = f"{self.config}-qrels" if self.config is not None else "default"
    if config == "default" and config not in self.dataset_configs:
        if "qrels" in self.dataset_configs:
            config = "qrels"
        else:
            raise ValueError(
                "No qrels or default config found. Please specify a valid config or ensure the dataset has qrels."
            )

    qrels_ds = self._load_dataset_split(config)
    qrels_ds = qrels_ds.select_columns(["query-id", "corpus-id", "score"])

    qrels_ds = qrels_ds.map(
        lambda batch: {"score": [_clip_to_uint16(score) for score in batch["score"]]},
        batched=True,
        desc="Clipping negative qrels scores to zero",
    )

    qrels_dict = {}
    for row in qrels_ds:
        qrels_dict.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels_dict


RetrievalDatasetLoader._load_qrels = _load_qrels_with_clipping  # noqa


def evaluate_model(
    model_name: str,
    output_dir: str = "results",
    task_names: Optional[List[str]] = None,
    pool_type: Optional[str] = None,
    encoding_method: Optional[str] = None,
    normalize: bool = True,
    general_instruction: str = "Given a query, retrieve relevant passages that answer the query",
    matryoshka_dim: Optional[int] = None,
    **kwargs,
):
    """
    Evaluate an embedding model on academic tasks using latest MTEB API.

    Args:
        model_name: HuggingFace model name or path
        output_dir: Directory to save results
        task_names: List of task names (default: all academic tasks)
        pool_type: Custom pooling strategy (None=auto, or 'cls'/'avg'/'last'/'weightedavg')
        encoding_method: Custom encoding method (None=auto)
        normalize: Whether to L2 normalize embeddings
        general_instruction: Instruction for instruction-based models
        matryoshka_dim: Dimension to truncate embeddings to
        **kwargs: Additional arguments for mteb.evaluate()
    """
    # Use all academic tasks if none specified
    if task_names is None:
        task_names = ACADEMIC_TASKS

    print(f"Loading model: {model_name}")

    # Extract additional parameters
    max_length = kwargs.pop("max_length", 4096)
    batch_size = kwargs.pop("batch_size", 128)

    # Use base embedding model if pooling or encoding is specified, or for better control
    if pool_type is not None or encoding_method is not None or model_name.count("/") > 1:
        print("Using custom embedding model with full control")
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
    else:
        # Try default loading methods
        try:
            # Try mteb.get_model() first - latest API
            model = mteb.get_model(model_name)
            print("Model loaded successfully with mteb.get_model()")
        except Exception as e:
            print(f"mteb.get_model() failed: {e}")

            # Fallback to SentenceTransformer if available
            print("Falling back to SentenceTransformer...")
            try:
                model = SentenceTransformer(model_name, trust_remote_code=True)
                print("Model loaded successfully with SentenceTransformer")
            except Exception as e2:
                print(f"Failed to load model with SentenceTransformer: {e2}")
                raise

    print(f"Loading tasks: {task_names}")
    try:
        # Use mteb.get_tasks() - latest API
        tasks = mteb.get_tasks(tasks=task_names)
    except Exception as e:
        print(f"Failed to load tasks: {e}")
        raise

    # Create output directory
    if matryoshka_dim:
        output_dir = f"{output_dir}/dim_{matryoshka_dim}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(tasks)} tasks...")

    # Run evaluation using mteb.evaluate() - latest API
    encode_kwargs = kwargs.pop("encode_kwargs", {})
    encode_kwargs["batch_size"] = batch_size

    evaluation = mteb.evaluate(
        model=model,  # type: ignore
        tasks=tasks,
        encode_kwargs=encode_kwargs,
        prediction_folder=str(output_path),
        **kwargs,
    )

    # Print summary
    print_evaluation_summary(evaluation, output_path)

    return evaluation


def print_evaluation_summary(evaluation, output_path):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for task_result in evaluation.task_results:
        print(f"\nTask: {task_result.task_name}")
        print("-" * 40)

        scores_by_split = task_result.scores
        split_scores = scores_by_split.get("test")
        if not split_scores:
            split_scores = next(iter(scores_by_split.values()), [])

        metrics = split_scores[0] if split_scores else {}

        if not metrics:
            print("  No scores available")
            continue

        if "ndcg_at_10" in metrics:
            print(f"  NDCG@10: {metrics['ndcg_at_10']:.4f}")
            if "map_at_10" in metrics:
                print(f"  MAP@10:  {metrics['map_at_10']:.4f}")
            if "recall_at_10" in metrics:
                print(f"  Recall@10: {metrics['recall_at_10']:.4f}")
        elif "accuracy" in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            if "f1" in metrics:
                print(f"  F1: {metrics['f1']:.4f}")
        elif "map" in metrics:
            print(f"  MAP: {metrics['map']:.4f}")
            if "mrr" in metrics:
                print(f"  MRR: {metrics['mrr']:.4f}")
        else:
            for key, value in list(metrics.items())[:3]:
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on academic MTEB tasks (using latest MTEB API)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., 'sentence-transformers/all-MiniLM-L6-v2')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results (default: 'results')",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=ACADEMIC_TASKS,
        help=f"Specific tasks to evaluate (default: all tasks). Choices: {', '.join(ACADEMIC_TASKS)}",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for encoding (default: 32)"
    )
    parser.add_argument(
        "--pool_type",
        type=str,
        default=None,
        choices=["cls", "avg", "last", "weightedavg"],
        help="Custom pooling strategy (default: None, auto-detect). Options: cls, avg, last, weightedavg",
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
        help="L2 normalize embeddings when using custom pooling (default: True)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization (default: 4096)",
    )
    parser.add_argument(
        "--general_instruction",
        type=str,
        default="Given a query, retrieve relevant passages that answer the query",
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

    # Run evaluation with latest MTEB API
    evaluate_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        task_names=args.tasks,
        batch_size=args.batch_size,
        pool_type=args.pool_type,
        encoding_method=args.encoding_method,
        normalize=args.normalize,
        max_length=args.max_length,
        general_instruction=args.general_instruction,
        matryoshka_dim=args.matryoshka_dim,
    )


if __name__ == "__main__":
    main()
