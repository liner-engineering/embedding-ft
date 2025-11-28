import re
from typing import Dict, List, Optional

import torch.distributed as dist
from transformers import PreTrainedTokenizerBase
from utils import TaskDescription, get_instruct_template


def is_main_process():
    """Check if current process is the main process (rank 0)"""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class E5EmbeddingDataTokenizer:
    """
    Tokenize data using E5EmbeddingTokenizer.

    :param tokenizer: PreTrainedTokenizerBase. Tokenizer
    :param max_length: Optional[int]. Specify max length
    :param task_type: Optional[str]. Specify task type. Default None
    :param end_with_eos: bool. Specify whether ends with the eos token. Default False.
    :param extra_columns: Optional[List[str]]. Specify extra columns. Default None.

    Example::

            from datasets import load_dataset
            import E5EmbeddingTokenizer

            # define dataset
            ds = load_dataset('your_dataset')

            # tokenize data
            train_ds = ds['train'].shuffle().map(E5EmbeddingTokenizer(tokenizer, max_length, task_type, True), num_proc=8)
            valid_ds = ds['validation'].map(E5EmbeddingTokenizer(tokenizer, max_length, task_type", True), num_proc=8)

    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        task_type: Optional[str] = None,
        end_with_eos: bool = False,
        extra_columns: Optional[List[str]] = None,
        instruction_type: Optional[str] = None,
        general_instruction: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extra_columns = extra_columns

        self.query_placeholder = "text"
        self.passage_placeholder = "passage"
        self.extra_placeholder_defaults: Dict[str, str] = {}

        self.query_prompt_template_tok = None
        self.query_prompt_template = None
        self.passage_prompt_template_tok = None
        self.passage_prompt_template = None

        self.end_with_eos = end_with_eos
        self.task_type = task_type
        template_placeholders = ["condition", "text", "passage", "content", "title"]
        if instruction_type == "task_specific":
            if task_type and task_type.lower() in list(
                map(str.lower, TaskDescription.list_task_types())
            ):
                self.query_prompt_template = get_instruct_template(task_type)
                re_placeholder = re.compile(r"\{(%s)\}" % "|".join(template_placeholders))
                self.query_prompt_template_tok = self.tokenizer(
                    re_placeholder.sub("", self.query_prompt_template)
                )
                if is_main_process():
                    print(f"Using task-specific instruction: {self.query_prompt_template}")
            else:
                raise ValueError(
                    f"Invalid task type: {task_type}. Must be one of {TaskDescription.list_task_types()}"
                )
        elif instruction_type == "general":
            # general_instruction = "Given a query, retrieve relevant passages that are most relevant to the query"
            self.query_prompt_template = f"Instruct: {general_instruction}\nQuery: {{text}}"
            re_placeholder = re.compile(r"\{(%s)\}" % "|".join(template_placeholders))
            self.query_prompt_template_tok = self.tokenizer(
                re_placeholder.sub("", self.query_prompt_template)
            )
            if is_main_process():
                print(f"Using general instruction: {general_instruction}")

        elif instruction_type == "indicate_query" or instruction_type == "query_or_passage":
            text_placeholder = "{text}"
            self.query_prompt_template = f"query: {text_placeholder}"
            re_placeholder = re.compile(r"\{(%s)\}" % "|".join(template_placeholders))
            self.query_prompt_template_tok = self.tokenizer(
                re_placeholder.sub("", self.query_prompt_template)
            )
            if is_main_process():
                print(f"Using indicate_query instruction: {self.query_prompt_template}")

            if instruction_type == "query_or_passage":
                self.passage_prompt_template = f"passage: {text_placeholder}"
                self.passage_prompt_template_tok = self.tokenizer(
                    re_placeholder.sub("", self.passage_prompt_template)
                )
                if is_main_process():
                    print(f"Using query_or_passage instruction: {self.passage_prompt_template}")

        elif instruction_type == "chat_query_passage":
            # text_placeholder = "{text}"
            self.query_prompt_template = (
                f"<|im_start|>system\n{general_instruction}<|im_end|>\n<|im_start|>query\n{{text}}"
            )
            re_placeholder = re.compile(r"\{(%s)\}" % "|".join(template_placeholders))
            self.query_prompt_template_tok = self.tokenizer(
                re_placeholder.sub("", self.query_prompt_template)
            )
            if is_main_process():
                print(f"Using chat_query_passage instruction - query: {self.query_prompt_template}")

            passage_placeholder = "{passage}"
            self.passage_prompt_template = f"<|im_start|>passage\n{passage_placeholder}"
            self.passage_prompt_template_tok = self.tokenizer(
                re_placeholder.sub("", self.passage_prompt_template)
            )
            if is_main_process():
                print(
                    f"Using chat_query_passage instruction - passage: {self.passage_prompt_template}"
                )

        elif instruction_type == "chat_user_assistant":
            self.query_prompt_template = (
                f"<|im_start|>system\n{general_instruction}<|im_end|>\n<|im_start|>user\n{{text}}"
            )
            re_placeholder = re.compile(r"\{(%s)\}" % "|".join(template_placeholders))
            self.query_prompt_template_tok = self.tokenizer(
                re_placeholder.sub("", self.query_prompt_template)
            )
            if is_main_process():
                print(
                    f"Using chat_user_assistant instruction - query: {self.query_prompt_template}"
                )

            passage_placeholder = "{passage}"
            self.passage_prompt_template = f"<|im_start|>assistant\n{passage_placeholder}"
            self.passage_prompt_template_tok = self.tokenizer(
                re_placeholder.sub("", self.passage_prompt_template)
            )
            if is_main_process():
                print(
                    f"Using chat_user_assistant instruction - passage: {self.passage_prompt_template}"
                )

        elif instruction_type == "embedding_gemma":
            self.query_placeholder = "content"
            self.query_prompt_template = "task: search result | query: {content}"
            re_placeholder = re.compile(r"\{(%s)\}" % "|".join(template_placeholders))
            self.query_prompt_template_tok = self.tokenizer(
                re_placeholder.sub("", self.query_prompt_template)
            )

            self.passage_placeholder = "content"
            self.passage_prompt_template = "title: none | text: {content}"
            self.passage_prompt_template_tok = self.tokenizer(
                re_placeholder.sub("", self.passage_prompt_template)
            )
            self.extra_placeholder_defaults["title"] = "none"

    def _apply_prompt_template(
        self,
        data: Dict,
        column: str,
        template: str,
        template_tok: Dict[str, List[int]],
        placeholder: str,
        extra_length: int,
        extra_placeholder: Dict[str, str],
    ) -> None:
        max_length = self.max_length - len(template_tok["input_ids"]) - extra_length

        def _format_text(text: str) -> str:
            tok = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
            )
            truncated_text = self.tokenizer.decode(tok["input_ids"])  # type: ignore
            format_kwargs = {key: str(val) for key, val in extra_placeholder.items()}
            format_kwargs[placeholder] = truncated_text
            return template.format(**format_kwargs)

        value = data[column]
        if isinstance(value, list):
            data[column] = [_format_text(item) for item in value]
        else:
            data[column] = _format_text(value)

    def _validate_prompt_tokens(
        self,
        toks: List,
        template_tok: Dict[str, List[int]],
    ) -> None:
        """Ensure every tokenized sequence starts with the expected prompt token."""
        expected_first_token = template_tok["input_ids"][0]

        def _ensure_match(tokenized) -> None:
            if tokenized.input_ids[0] != expected_first_token:
                raise RuntimeError(
                    f"something wrong when tokenizing prompt template. {tokenized.input_ids[0]} != {expected_first_token}"
                )

        for tokenized in toks:
            if isinstance(tokenized, list):
                for nested_tok in tokenized:
                    _ensure_match(nested_tok)
            else:
                _ensure_match(tokenized)

    def __call__(self, data: Dict) -> Dict:
        """Tokenize a sample that bundles a query with its positive and negative passages.

        Processing steps:

        1. Ensure the mandatory ``text``, ``positive``, and ``negative`` columns exist.
        2. Inject query/passage prompt templates when configured, trimming each string to
           honor ``max_length`` as well as any registered ``extra_columns`` values that
           appear as ``{column_name}`` placeholders in the templates.
        3. Tokenize each string (flattening lists of negatives) and collect the
           tokenizer outputs in order.
        4. Concatenate matching tokenizer fields (e.g., ``input_ids``, ``attention_mask``)
           and record provenance in ``seperate_ids`` so downstream consumers know which
           tokens came from which source: index 0 for the query, 1 for the positive, and
           increasing indices for each negative.

        Example output structure::

            {
                "input_ids": [101, ..., 102, 101, ..., 102, ...],
                "attention_mask": [1, 1, 1, ...],
                "seperate_ids": [0, 0, 0, 1, 1, 1, 2, 2, ...],
                "extra": {"end_with_eos": False},
                "task_type": "retrieval"
            }

        Args:
            data (Dict): Input mapping that must include the ``text`` (query), ``positive``
                (relevant passage), and ``negative`` (hard negatives) keys. ``negative`` may
                be a single string or a list. Any columns declared in ``extra_columns`` can
                be provided here as additional template placeholders.

        Returns:
            Dict: Tokenizer outputs with concatenated sequences plus ``seperate_ids``, an
                ``extra`` metadata block, and the configured ``task_type``.
        """
        if not ("text" in data and "positive" in data and "negative" in data):
            raise NotImplementedError("must include three columns: `text`, `positive`, `negative`")

        text_columns = ["text", "positive", "negative"]

        extra_length = 0
        extra_placeholder = {}
        if self.extra_columns is not None:
            for key, val in data.items():
                if key not in self.extra_columns:
                    continue
                extra_placeholder[key] = val
                extra_length += len(
                    self.tokenizer(val, add_special_tokens=False)["input_ids"]  # type: ignore
                )
        if self.end_with_eos:
            extra_length += 1

        for key, val in self.extra_placeholder_defaults.items():
            extra_placeholder.setdefault(key, val)

        # set prompt template to query column
        if self.query_prompt_template_tok is not None and self.query_prompt_template is not None:
            self._apply_prompt_template(
                data=data,
                column="text",
                template=self.query_prompt_template,
                template_tok=self.query_prompt_template_tok,  # type: ignore
                placeholder=self.query_placeholder,
                extra_length=extra_length,
                extra_placeholder=extra_placeholder,
            )

        if (
            self.passage_prompt_template is not None
            and self.passage_prompt_template_tok is not None
        ):
            for passage_column in ["positive", "negative"]:
                self._apply_prompt_template(
                    data=data,
                    column=passage_column,
                    template=self.passage_prompt_template,
                    template_tok=self.passage_prompt_template_tok,  # type: ignore
                    placeholder=self.passage_placeholder,
                    extra_length=extra_length,
                    extra_placeholder=extra_placeholder,
                )

        # tokenize all text columns
        toks = []
        for text_column in text_columns:
            if isinstance(data[text_column], list):
                for idx, text in enumerate(data[text_column]):
                    toks.append(self.tokenizer(text, max_length=self.max_length, truncation=True))
            else:
                toks.append(
                    self.tokenizer(data[text_column], max_length=self.max_length, truncation=True)
                )
            # toks.append(
            #     self.tokenizer(data[text_column], max_length=self.max_length, truncation=True)
            # )

        # bad data - validate query tokens (only the first token which is 'text' column)
        if self.query_prompt_template_tok is not None:
            self._validate_prompt_tokens([toks[0]], self.query_prompt_template_tok)  # type: ignore

        # bad data - validate passage tokens (positive and negative columns)
        if self.passage_prompt_template_tok is not None:
            self._validate_prompt_tokens(toks[1:], self.passage_prompt_template_tok)  # type: ignore

        # combine tokenized data
        combined_tok = {}
        seperate_ids = []
        for idx, tok in enumerate(toks):
            for key, val in tok.items():
                if idx == 0:
                    combined_tok[key] = val
                else:
                    combined_tok[key] += val
                if key == "input_ids":
                    seperate_ids += [idx] * len(val)

        combined_tok["seperate_ids"] = seperate_ids
        combined_tok["extra"] = {"end_with_eos": self.end_with_eos}
        combined_tok["task_type"] = self.task_type
        return combined_tok
