"""
QASA Evaluation Script using BEIR
Evaluates embedding models on QASA dataset using BEIR's DenseRetrievalExactSearch.

This script:
1. Loads paper section data from parquet
2. Prepares corpus and queries in BEIR format
3. Uses BEIR's DenseRetrievalExactSearch for evaluation
4. Computes standard retrieval metrics (NDCG, MAP, Recall, etc.)
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from embedding.common.base_model import BaseEmbeddingModel
from embedding.evaluate.common_evaluation import (
    build_results_dict,
    print_evaluation_results,
    save_evaluation_results,
)


def prepare_beir_data(
    df_corpus: pd.DataFrame,
    queries_data: list,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], Dict[str, str]]:
    """
    Prepare data in BEIR format for QASA dataset.

    Args:
        df_corpus: DataFrame with corpus sections (columns: title, corpusid, arxiv_id, section_idx, section, num_chars)
        queries_data: List of query dictionaries loaded from JSONL (keys: input, answer, gold_ctxs, ctxs)

    Returns:
        Tuple of (corpus, queries, qrels, doc_id_to_canonical) in BEIR format
        - corpus: Dict[doc_id, Dict[title, text]]
        - queries: Dict[query_id, query_text]
        - qrels: Dict[query_id, Dict[doc_id, relevance_score]]
        - doc_id_to_canonical: Dict[doc_id, canonical_doc_id] - maps duplicate title docs to first doc
    """
    # Prepare corpus with title-based mapping (following evaluate_qasa_pinecone.py logic)
    corpus = {}
    title_to_doc_ids = {}  # Map from section title to list of doc_ids (handle duplicate titles)

    for idx, row in df_corpus.iterrows():
        doc_id = f"doc_{idx}"
        section_text = str(row["section"])
        section_title = str(row["title"])

        corpus[doc_id] = {
            "title": section_title,
            "text": section_text,
        }

        # Map section title to doc_ids for matching with ctxs (support multiple docs with same title)
        normalized_title = section_title.lower().strip()
        if normalized_title not in title_to_doc_ids:
            title_to_doc_ids[normalized_title] = []
        title_to_doc_ids[normalized_title].append(doc_id)

    print(f"Created corpus with {len(corpus)} documents")
    print(f"Number of unique titles: {len(title_to_doc_ids)}")
    print(f"Sample section titles: {list(title_to_doc_ids.keys())[:3]}")

    # Prepare queries and qrels
    queries = {}
    qrels = {}

    for query_idx, query_item in enumerate(queries_data):
        query_id = f"query_{query_idx}"
        queries[query_id] = str(query_item["input"])

        # Prepare qrels (ground truth relevance)
        if "ctxs" in query_item and query_item["ctxs"]:
            qrels[query_id] = {}

            # Use gold_ctxs indices to get relevant contexts
            gold_ctxs_indices = query_item.get("gold_ctxs", [])

            for ctx_idx in gold_ctxs_indices:
                if ctx_idx < len(query_item["ctxs"]):
                    ctx = query_item["ctxs"][ctx_idx]
                    # Match by title (same as evaluate_qasa_pinecone.py)
                    gt_title = ctx["title"].lower().strip()

                    # Find matching doc_ids by title
                    if gt_title in title_to_doc_ids:
                        # Mark only the first document with this title as relevant
                        # This way, retrieving any one of the duplicate title rows counts as full recall
                        doc_id = title_to_doc_ids[gt_title][0]
                        qrels[query_id][doc_id] = 1  # Binary relevance

    print(f"Created {len(queries)} queries")
    print(f"Created qrels for {len(qrels)} queries")

    # Create doc_id to canonical_doc_id mapping
    # For each title, map all doc_ids to the first doc_id (canonical)
    doc_id_to_canonical = {}
    for title, doc_ids in title_to_doc_ids.items():
        canonical_doc_id = doc_ids[0]  # First doc_id is canonical
        for doc_id in doc_ids:
            doc_id_to_canonical[doc_id] = canonical_doc_id

    print(f"Created doc_id_to_canonical mapping with {len(doc_id_to_canonical)} entries")

    return corpus, queries, qrels, doc_id_to_canonical


def normalize_retrieval_results(
    results: Dict[str, Dict[str, float]],
    doc_id_to_canonical: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    Normalize retrieval results by mapping doc_ids to canonical doc_ids.

    For documents with the same title, all doc_ids are mapped to the canonical (first) doc_id.
    This ensures that retrieving any duplicate title document counts as finding the canonical one.

    Args:
        results: Dict[query_id, Dict[doc_id, score]] - raw retrieval results
        doc_id_to_canonical: Dict[doc_id, canonical_doc_id] - mapping to canonical docs

    Returns:
        Normalized results with doc_ids replaced by canonical doc_ids.
        If multiple duplicates are retrieved, keeps the highest score.
    """
    normalized_results = {}

    for query_id, doc_scores in results.items():
        normalized_scores = {}

        for doc_id, score in doc_scores.items():
            # Map to canonical doc_id
            canonical_doc_id = doc_id_to_canonical.get(doc_id, doc_id)

            # Keep highest score if multiple duplicates are retrieved
            if canonical_doc_id in normalized_scores:
                normalized_scores[canonical_doc_id] = max(
                    normalized_scores[canonical_doc_id], score
                )
            else:
                normalized_scores[canonical_doc_id] = score

        normalized_results[query_id] = normalized_scores

    return normalized_results


def evaluate_qasa(
    model_name: str,
    corpus_path: str,
    query_path: str,
    output_dir: str = "results",
    pool_type: Optional[str] = None,
    encoding_method: Optional[str] = None,
    normalize: bool = True,
    max_length: int = 4096,
    batch_size: int = 128,
    score_function: str = "cos_sim",
    general_instruction: str = "Given a query, retrieve relevant passages that answer the query",
    matryoshka_dim: Optional[int] = None,
):
    """
    Evaluate embedding model on QASA dataset using BEIR's DenseRetrievalExactSearch.

    Args:
        model_name: HuggingFace model name or path
        corpus_path: Path to QASA corpus parquet file (e.g., qasa_section.parquet)
        query_path: Path to QASA query JSONL file (e.g., qasa_data_qasa_test.jsonl)
        output_dir: Directory to save results
        pool_type: Custom pooling strategy (None=auto, or 'cls'/'avg'/'last'/'weightedavg')
        encoding_method: Custom encoding method (None=auto)
        normalize: Whether to L2 normalize embeddings
        max_length: Maximum sequence length
        batch_size: Batch size for encoding
        score_function: Scoring function for Exact Search ('cos_sim' or 'dot')
        general_instruction: Instruction for instruction-based models
    """
    import json

    df_corpus = pd.read_parquet(corpus_path)
    queries_data = []
    query_file = Path(query_path)
    with query_file.open("r") as f:
        for line in f:
            queries_data.append(json.loads(line.strip()))

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
    corpus, queries, qrels, doc_id_to_canonical = prepare_beir_data(
        df_corpus=df_corpus,
        queries_data=queries_data,
    )

    # Run retrieval (without evaluation yet)
    print("\nRunning retrieval...")
    k_values = [1, 3, 5, 10, 20, 100]
    exact_search = DRES(model, batch_size=batch_size)
    retriever = EvaluateRetrieval(exact_search, score_function=score_function, k_values=k_values)

    # Get raw retrieval results
    results = retriever.retrieve(corpus, queries)

    # Normalize results: map duplicate title doc_ids to canonical doc_ids
    print("\nNormalizing retrieval results (handling duplicate titles)...")
    normalized_results = normalize_retrieval_results(results, doc_id_to_canonical)

    # Evaluate with normalized results
    print("\nComputing metrics...")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, normalized_results, k_values)

    if results is None:
        return None

    # Print results
    print_evaluation_results(
        ndcg=ndcg,
        _map=_map,
        recall=recall,
        precision=precision,
        title="EVALUATION RESULTS - QASA",
    )

    # Prepare and save results
    metadata = {
        "model_name": model_name,
        "corpus_path": corpus_path,
        "query_path": query_path,
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

    output_filename = "qasa_exact_search_results.json"
    if matryoshka_dim:
        output_filename = f"qasa_exact_search_results_dim{matryoshka_dim}.json"

    save_evaluation_results(
        output_dir=output_dir,
        output_filename=output_filename,
        results_dict=results_dict,
    )

    return results_dict


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on QASA dataset using BEIR"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="qasa_section.parquet",
        help="Path to QASA corpus parquet file (e.g., qasa_section.parquet)",
    )
    parser.add_argument(
        "--query_path",
        type=str,
        default="qasa_data_qasa_test.jsonl",
        help="Path to QASA query JSONL file (e.g., qasa_data_qasa_test.jsonl)",
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
        default="Given a query, retrieve relevant passages that answer the query",
        help="General instruction for instruction-based encoding",
    )
    parser.add_argument(
        "--matryoshka_dim",
        type=int,
        default=None,
        help="Dimension to truncate embeddings to (for MRL evaluation)",
    )

    args = parser.parse_args()

    # Print configuration
    print("Configuration:")
    print(json.dumps(vars(args), indent=2))

    # Run evaluation
    evaluate_qasa(
        model_name=args.model_name,
        corpus_path=args.corpus_path,
        query_path=args.query_path,
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
