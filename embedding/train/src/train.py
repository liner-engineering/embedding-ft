# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from pathlib import Path

import datasets
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, SchedulerType, TrainingArguments
from utils import load_dataset_from_config

from embedding.common.heads import save_embeddinggemma_modules
from src.component.collator import E5EmbeddingDataCollator
from src.component.tokenizer import E5EmbeddingDataTokenizer
from src.component.trainer import E5EmbeddingTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Training a PEFT model for Sematic Search task")
    parser.add_argument("--dataset_config", type=str, default=None, help="dataset config")
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--instruction_type",
        type=str,
        required=True,
        choices=[
            "task_specific",
            "general",
            "indicate_query",
            "chat_user_assistant",
            "chat_query_passage",
            "query_passage",
            "embedding_gemma",
        ],
    )

    parser.add_argument(
        "--general_instruction",
        type=str,
        default="Given a query, retrieve relevant passages that answer the query",
        help="Instruction for the general instruction type",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging Steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--sanity_test",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Whether to use peft.",
    )
    parser.add_argument(
        "--continual_learning",
        action="store_true",
        help="Whether to enable continual learning.",
    )
    parser.add_argument(
        "--homogeneous_batching",
        action="store_true",
        help="Whether to homogeneous_batching.",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="Whether to group by length.",
    )
    parser.add_argument(
        "--matryoshka_dims",
        type=str,
        default=None,
        help=(
            "Comma-separated embedding dimensions to enable Matryoshka Representation Learning. "
            "If unset, the standard single-scale loss is used."
        ),
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--pooling_strategy",
        type=str,
        default="last",
        choices=["last", "avg", "cls"],
        help="Pooling strategy to use for getting sentence embeddings",
    )

    return parser.parse_args()


def main():
    #######
    # Setup
    #######
    args = parse_args()
    if args.matryoshka_dims:
        args.matryoshka_dims = [
            int(dim.strip()) for dim in args.matryoshka_dims.split(",") if dim.strip()
        ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator_kwargs = {"gradient_accumulation_steps": args.gradient_accumulation_steps}
    if args.with_tracking:
        accelerator_kwargs["log_with"] = args.report_to
        accelerator_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(**accelerator_kwargs)

    if accelerator.is_main_process:
        print(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    #######
    # Handle the output directory creation
    #######
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    #######
    # Load datasets and tokenizer and preprocess datasets
    #######
    # tokenizer
    if accelerator.is_main_process:
        print("#### Instruction Type: %s ####" % args.instruction_type)
    tokenizer_kwargs = {"trust_remote_code": True}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=False, **tokenizer_kwargs
        )
    except (OSError, ValueError) as err:
        logging.warning(
            "Falling back to the fast tokenizer because the slow tokenizer could not be loaded: %s",
            err,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True, **tokenizer_kwargs
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset download and preprocessing
    if args.sanity_test:
        dataset = load_dataset_from_config("config/dataset_config/test1.yaml")
        args.per_device_train_batch_size = min(2, args.per_device_train_batch_size)
        args.gradient_accumulation_steps = min(2, args.gradient_accumulation_steps)
        args.save_steps = min(2, args.save_steps)
    else:
        dataset = load_dataset_from_config(args.dataset_config)

    # concatenate all dataset into one
    train_dataset_list: list[datasets.Dataset] = []
    for ds in dataset:
        data = ds["data"]
        task_type = ds["task_type"]
        train_dataset_list.append(
            data.map(
                E5EmbeddingDataTokenizer(
                    tokenizer,
                    args.max_length,
                    task_type,
                    end_with_eos=True,
                    instruction_type=args.instruction_type,
                    general_instruction=args.general_instruction,
                ),
                num_proc=32,
            )
        )

    num_sep_ids = len(
        set(train_dataset_list[0][0]["seperate_ids"])
    )  # number of negative samples + 2 (text, positive)
    if accelerator.is_main_process:
        print("### num_sep_ids:", num_sep_ids)

    train_ds = datasets.concatenate_datasets(train_dataset_list)
    train_ds = train_ds.shuffle()
    # train_ds = dataset.shuffle().map(E5EmbeddingDataTokenizer(tokenizer, args.max_length, "commonsense", True), num_proc=8)

    if accelerator.is_main_process:
        print(f"Length of the training set: {len(train_ds)}.")
        # Log a few random samples from the training set:
        for i in range(3):
            print(f"Sample {i} of the training set: {train_ds[i]}.")

    if args.precision == "fp16":
        fp16 = True
        bf16 = False
    elif args.precision == "bf16":
        fp16 = False
        bf16 = True
    else:
        fp16 = False
        bf16 = False

    #########################
    # Instantiate E5EmbeddingTrainer
    #########################
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},  # type: ignore
        warmup_steps=args.num_warmup_steps,
        learning_rate=args.learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_strategy="steps",  # TODO: change to args.logging_strategy
        logging_steps=args.logging_steps,
        save_strategy="steps",  # TODO: change to args.save_strategy
        save_steps=args.save_steps,
        load_best_model_at_end=False,
        save_total_limit=args.save_total_limit,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        label_names=["seperate_ids", "extra", "task_type"],
    )

    # initialize trainer
    trainer = E5EmbeddingTrainer(
        model_name_or_path=args.model_name_or_path,
        use_peft=args.use_peft,
        train_dataset=train_ds,
        eval_dataset=None,
        args=training_args,
        continual_learning=args.continual_learning,
        homogeneous_batching=args.homogeneous_batching,
        group_by_length=args.group_by_length,
        num_sep_ids=num_sep_ids,
        matryoshka_dims=args.matryoshka_dims,
        pooling_strategy=args.pooling_strategy,
        data_collator=E5EmbeddingDataCollator(
            tokenizer, return_tensors="pt", max_length=args.max_length
        ),
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("peft_semantic_search", experiment_config)

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_ds)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_steps}")

    ###############
    # Training loop
    ###############
    train_result = trainer.train(args.resume_from_checkpoint)
    trainer.model.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)
    if accelerator.is_main_process:
        model_to_save = trainer.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module  # type: ignore[assignment]
        embedding_head = getattr(model_to_save, "embedding_head", None)
        if embedding_head is not None:
            save_embeddinggemma_modules(embedding_head, args.output_dir)

    metrics = train_result.metrics
    max_train_samples = len(train_ds) * args.num_train_epochs
    metrics["train_samples"] = min(max_train_samples, len(train_ds))
    if accelerator.is_main_process:
        print("*** Training complete ***")

    ###############
    # Save state
    ###############
    # Use accelerator.print to print only on the main process.
    accelerator.wait_for_everyone()
    if args.output_dir is not None and accelerator.is_main_process:
        # peft_model_id = f"{args.dataset_name}_{args.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_")
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
