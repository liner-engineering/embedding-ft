"""Base embedding model class with shared logic."""

import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from FlagEmbedding import BGEM3FlagModel
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import BatchedInput, PromptType
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .config import ENCODING_METHOD, POOLING_METHOD
from .heads import EmbeddingGemmaProjectionHead
from .utils import cos_sim, create_batch_dict, move_to_cuda, pool


class BaseEmbeddingModel(AbsEncoder):
    """
    Base embedding model with configurable pooling and encoding methods.
    Compatible with both BEIR (DenseRetrievalFaissSearch/ExactSearch) and MTEB evaluation.

    This class provides encoding methods (encode_queries, encode_corpus, encode)
    that can be used with various BEIR retrieval backends (FAISS, Exact Search, etc.)
    """

    def __init__(
        self,
        model_name: str,
        pool_type: Optional[str] = None,
        encoding_method: Optional[str] = None,
        max_length: int = 4096,
        batch_size: int = 32,
        general_instruction: str = "Given a query, retrieve relevant passages that answer the query",
        normalize: bool = True,
        matryoshka_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize base embedding model.

        Args:
            model_name: HuggingFace model name or path
            pool_type: Pooling strategy (None=auto, 'cls', 'avg', 'last', 'weightedavg')
            encoding_method: Encoding method (None=auto, 'no-prefix', 'query_or_passage', 'instruction', etc.)
            max_length: Maximum sequence length (default: 4096)
            batch_size: Batch size for encoding
            general_instruction: General instruction for instruction-based models
            normalize: Whether to L2 normalize embeddings (default: True)
            matryoshka_dim: Dimension for Matryoshka Representation Learning (truncate embeddings)
            **kwargs: Additional arguments for compatibility
        """
        self.model_name_or_path = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.general_instruction = general_instruction
        self.normalize = normalize
        self.matryoshka_dim = matryoshka_dim
        self.prompt = None  # For dynamic prompt setting
        self.kwargs = kwargs
        self.embedding_head: Optional[torch.nn.Module] = None
        self.encoder_dtype = torch.float16

        # MTEB compatibility - create model metadata
        meta_name = model_name
        meta_revision = None
        if matryoshka_dim:
            meta_name = f"{model_name}_dim_{matryoshka_dim}"
            meta_revision = f"dim_{matryoshka_dim}"

        self.mteb_model_meta = ModelMeta(
            name=meta_name,
            loader=None,  # Not needed for direct instantiation
            revision=meta_revision,
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],  # Required field
            similarity_fn_name=None,
            use_instructions=None,
            training_datasets=None,
        )

        # BEIR compatibility attributes
        # These are used when BaseEmbeddingModel is used with BEIR's evaluation frameworks
        self.score_functions = {"cos_sim": cos_sim}
        self.score_function_desc = {"cos_sim": "Cosine Similarity"}

        # Auto-detect encoding method and pooling if not specified
        self.encoding_method = encoding_method or ENCODING_METHOD.get(model_name.split("/")[-1])
        self.pool_type = pool_type or POOLING_METHOD.get(model_name.split("/")[-1])

        assert self.encoding_method, (
            f"Encoding method is not defined for {model_name}. "
            "Please provide desired encoding method."
        )

        # BGE-M3 doesn't need pool_type
        if model_name.split("/")[-1] != "bge-m3":
            assert self.pool_type, (
                f"Pooling method is not defined for {model_name}. "
                "Please provide desired pooling method."
            )

        print(f"### encoding method: {self.encoding_method}")
        if self.pool_type:
            print(f"### pool type: {self.pool_type}")

        # Check if model path is local
        is_local_path = Path(self.model_name_or_path).exists()

        # Handle BGE-M3 separately
        if self.model_name_or_path.split("/")[-1] == "bge-m3":
            self.encoder = BGEM3FlagModel(self.model_name_or_path, use_fp16=True)
            self.gpu_count = torch.cuda.device_count()
            self.tokenizer = None  # Tokenizer is included in BGEM3FlagModel
        else:
            # Check if model is decoder-only (needs use_cache=False for DataParallel)
            model_name_lower = self.model_name_or_path.lower()
            is_decoder_model = any(
                name in model_name_lower for name in ["qwen", "llama", "mistral", "gemma"]
            )

            if self.encoding_method == "embedding_gemma":
                self.encoder_dtype = torch.bfloat16

            # Load model with appropriate configuration
            if is_decoder_model:
                # For decoder-only models, disable KV cache to prevent OOM with DataParallel
                self.encoder = AutoModel.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=self.encoder_dtype,
                    trust_remote_code=True,
                    use_cache=False,  # Disable KV cache for decoder models
                    local_files_only=is_local_path,
                )
            else:
                # For encoder-only models (BERT, GTE, etc.), don't use use_cache parameter
                self.encoder = AutoModel.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=self.encoder_dtype,
                    trust_remote_code=True,
                    local_files_only=is_local_path,
                )

            self.gpu_count = torch.cuda.device_count()
            if self.gpu_count > 1:
                self.encoder = torch.nn.DataParallel(self.encoder)

            self.encoder.cuda()
            self.encoder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local_path,
        )
        if self.encoding_method == "embedding_gemma":
            head = EmbeddingGemmaProjectionHead(self.model_name_or_path)
            ref_dtype = next(self.encoder.parameters()).dtype  # type: ignore[union-attr]
            head = head.to(dtype=ref_dtype)
            if torch.cuda.is_available():
                head = head.cuda()
                if self.gpu_count > 1:
                    head = torch.nn.DataParallel(head)
            head.eval()
            self.embedding_head = head

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries with query-specific prefix.

        Args:
            queries: List of query strings
            **kwargs: Additional arguments (ignored for compatibility with BEIR)
        """
        _ = kwargs  # Suppress unused argument warning - for BEIR compatibility
        if self.encoding_method == "instruction":
            input_texts = [f"Instruct: {self.general_instruction}\nQuery: {q}" for q in queries]
        elif self.encoding_method == "chat_user_assistant":
            input_texts = [
                f"<|im_start|>system\n{self.general_instruction}<|im_end|>\n<|im_start|>user\n{q}"
                for q in queries
            ]
        elif self.encoding_method == "chat_query_passage":
            input_texts = [
                f"<|im_start|>system\n{self.general_instruction}<|im_end|>\n<|im_start|>query\n{q}"
                for q in queries
            ]
        elif self.encoding_method == "query_or_passage" or self.encoding_method == "query":
            input_texts = [f"query: {q}" for q in queries]
        elif self.encoding_method == "embedding_gemma":
            input_texts = [f"task: search result | query: {q}" for q in queries]
        else:
            input_texts = queries

        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        """Encode corpus with passage-specific prefix.

        Args:
            corpus: List of documents with 'title' and 'text' keys
            **kwargs: Additional arguments (ignored for compatibility with BEIR)
        """
        _ = kwargs  # Suppress unused argument warning - for BEIR compatibility
        input_texts = ["{}\n{}".format(doc.get("title", ""), doc["text"]).strip() for doc in corpus]
        if self.encoding_method == "chat_user_assistant":
            input_texts = [f"<|im_start|>assistant\n{t}" for t in input_texts]
        elif self.encoding_method == "chat_query_passage":
            input_texts = [f"<|im_start|>passage\n{t}" for t in input_texts]
        elif self.encoding_method == "query_or_passage":
            input_texts = [f"passage: {t}" for t in input_texts]
        elif self.encoding_method == "embedding_gemma":
            input_texts = [f"title: none | text: {doc['text']}" for doc in corpus]

        return self._do_encode(input_texts)

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Generic encode method for MTEB classification/clustering tasks.

        Args:
            sentences: List of sentences to encode

        Returns:
            numpy array of embeddings
        """
        if self.prompt:
            input_texts: List[str] = [self.prompt + s for s in sentences]
        else:
            input_texts = sentences

        return self._do_encode(input_texts)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode method required by MTEB's AbsEncoder interface.

        Args:
            inputs: DataLoader containing batched inputs
            task_metadata: Task metadata
            hf_split: HuggingFace split
            hf_subset: HuggingFace subset
            prompt_type: Prompt type (query/passage)
            **kwargs: Additional arguments

        Returns:
            numpy array of embeddings
        """
        # Suppress unused argument warnings - required by abstract method
        _ = task_metadata, hf_split, hf_subset, kwargs

        # Collect all data from the DataLoader
        all_data = []

        for batch in inputs:
            # MTEB sends batches as dicts with list values
            if isinstance(batch, dict):
                # Get batch size from any available field
                batch_size = len(next(iter(batch.values())))

                # Process each item in the batch
                for idx in range(batch_size):
                    item_data = {}

                    # Collect all fields for this index
                    for key, values in batch.items():
                        if isinstance(values, list) and idx < len(values):
                            item_data[key] = values[idx]

                    # Handle text field - MTEB already combines title+text for documents
                    # For documents: text = "title text" (already combined by MTEB)
                    # For queries: text = query text
                    if "text" in item_data:
                        text = item_data["text"]
                        # Handle empty text by using space as placeholder
                        if not text or (isinstance(text, str) and not text.strip()):
                            text = " "
                        all_data.append(
                            {
                                "text": text,
                                "title": item_data.get("title", ""),
                                "body": item_data.get("body", ""),
                                "query": item_data.get("query", ""),
                            }
                        )
                    else:
                        raise ValueError(f"No text field found in batch: {batch}")

            elif isinstance(batch, list):
                # Handle list inputs
                for item in batch:
                    if isinstance(item, str):
                        all_data.append({"text": item if item else " "})
                    elif isinstance(item, dict):
                        text = item.get("text", item.get("query", item.get("passage", " ")))
                        all_data.append({"text": text if text else " "})
                    else:
                        all_data.append({"text": str(item) if item else " "})

            elif isinstance(batch, str):
                # Single string
                all_data.append({"text": batch if batch else " "})

        # Extract texts based on prompt_type
        if prompt_type is not None and prompt_type.value == "query":
            texts = [item["text"] for item in all_data]
            return self.encode_queries(texts)

        if prompt_type is not None and prompt_type.value == "document":
            corpus = []
            for item in all_data:
                corpus.append({"title": item.get("title", ""), "text": item["text"]})
            return self.encode_corpus(corpus)

        texts = [item["text"] for item in all_data]
        return self.encode_sentences(texts)

    def set_prompt(self, prompt: Optional[str]):
        """
        Set a custom prompt for encoding.

        Args:
            prompt: Prompt string to prepend to sentences, or None to clear
        """
        self.prompt = prompt

    def set_matryoshka_dim(self, dim: Optional[int]):
        """
        Set the dimension for Matryoshka Representation Learning.

        Args:
            dim: Dimension to truncate embeddings to, or None to use full dimension
        """
        self.matryoshka_dim = dim

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        """Internal encoding method."""
        encoded_embeds = []
        batch_size = self.batch_size * self.gpu_count
        for start_idx in tqdm.tqdm(
            range(0, len(input_texts), batch_size), desc="encoding", mininterval=10
        ):
            batch_input_texts: List[str] = input_texts[start_idx : start_idx + batch_size]

            if self.model_name_or_path.split("/")[-1] == "bge-m3":
                # BGE-M3 uses its own encoding method
                embeds = self.encoder.encode(  # type: ignore
                    batch_input_texts,
                    batch_size=batch_size,
                    max_length=self.max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )["dense_vecs"]
                if self.normalize:
                    normalized_embeds = []
                    for emb in embeds:
                        normalized_embeds.append(emb / np.linalg.norm(emb))
                    encoded_embeds.append(np.array(normalized_embeds))
                else:
                    encoded_embeds.append(embeds)
            else:
                if self.tokenizer is None:
                    raise ValueError("Tokenizer is not initialized for this model")
                batch_dict = create_batch_dict(
                    self.tokenizer,
                    batch_input_texts,
                    always_add_eos=(self.pool_type == "last"),
                    max_length=self.max_length,
                )
                batch_dict = move_to_cuda(batch_dict)

                autocast_ctx = (
                    torch.amp.autocast("cuda", dtype=self.encoder_dtype)
                    if self.encoder_dtype in (torch.float16, torch.bfloat16)
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    # Check if model is decoder-only and add use_cache=False if needed
                    model_name_lower = self.model_name_or_path.lower()
                    is_decoder_model = any(
                        name in model_name_lower for name in ["qwen", "llama", "mistral", "gemma"]
                    )

                    if is_decoder_model:
                        # Disable KV cache for decoder-only models to prevent OOM with DataParallel
                        outputs = self.encoder(**batch_dict, use_cache=False)  # type: ignore
                    else:
                        outputs = self.encoder(**batch_dict)  # type: ignore

                    attention_mask = batch_dict["attention_mask"]  # type: ignore
                    if self.pool_type:
                        embeds = pool(
                            outputs.last_hidden_state,
                            attention_mask,  # type: ignore
                            self.pool_type,
                        )
                    else:
                        # Default to mean pooling if pool_type is None
                        embeds = pool(
                            outputs.last_hidden_state,
                            attention_mask,  # type: ignore
                            "avg",
                        )
                    if self.embedding_head is not None:
                        embeds = self.embedding_head(embeds)

                    if self.matryoshka_dim:
                        embeds = embeds[..., : self.matryoshka_dim]

                    if self.normalize:
                        norm = torch.linalg.norm(embeds, ord=2, dim=-1, keepdim=True)
                        norm = torch.clamp(norm, min=1e-12)
                        embeds = embeds / norm
                    encoded_embeds.append(embeds.cpu().numpy())
        return np.concatenate(encoded_embeds, axis=0)
