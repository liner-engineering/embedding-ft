import json
from pathlib import Path
from typing import List, Optional

import datasets
import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, Trainer
from transformers.utils import is_datasets_available

from embedding.common.heads import (
    EmbeddingGemmaProjectionHead,
    save_embeddinggemma_modules,
)
from embedding.common.hf_patch import apply_zero_size_embedding_guard
from embedding.common.utils import pool
from src.component.loss import E5EmbeddingLoss
from src.component.sampler import DataHomogeneousSampler, GroupByLengthSampler


class E5EmbeddingTrainer(Trainer):
    def __init__(
        self,
        model_name_or_path: str,
        num_sep_ids: int,
        use_peft: bool = False,
        continual_learning: bool = False,
        homogeneous_batching: bool = False,
        group_by_length: bool = False,
        matryoshka_dims: Optional[List[int]] = None,
        pooling_strategy: str = "last",
        **kwargs,
    ):
        apply_zero_size_embedding_guard()
        self.num_sep_ids = num_sep_ids
        self.matryoshka_dims = matryoshka_dims
        model_basename = Path(model_name_or_path).name.lower()
        self.model_dtype = torch.float16
        if "embeddinggemma" in model_basename:
            self.model_dtype = torch.bfloat16

        if use_peft:
            if continual_learning:
                config = PeftConfig.from_pretrained(model_name_or_path)
                base_model_path = config.base_model_name_or_path or model_name_or_path

                # Load base config to check for specific arguments
                base_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
                model_kwargs = {
                    "torch_dtype": self.model_dtype,
                    "trust_remote_code": True,
                }
                if getattr(base_config, "model_type", None) == "gte":
                    model_kwargs["unpad_inputs"] = False
                    model_kwargs["use_memory_efficient_attention"] = False

                self.model = AutoModel.from_pretrained(base_model_path, **model_kwargs)
                self.model = PeftModel.from_pretrained(
                    model=self.model, model_id=model_name_or_path, is_trainable=True
                )
                if kwargs["args"].gradient_checkpointing:
                    enable_grads = getattr(self.model, "enable_input_require_grads", None)
                    if callable(enable_grads):
                        enable_grads()
            else:
                config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
                model_kwargs = {
                    "torch_dtype": self.model_dtype,
                    "trust_remote_code": True,
                }
                if getattr(config, "model_type", None) == "gte":
                    model_kwargs["unpad_inputs"] = False
                    model_kwargs["use_memory_efficient_attention"] = False

                self.model = AutoModel.from_pretrained(
                    model_name_or_path, config=config, **model_kwargs
                )

                peft_config_path = Path("config/lora_config/lora.json")
                peft_config = LoraConfig(**json.loads(peft_config_path.read_text()))
                peft_config.base_model_name_or_path = getattr(
                    config, "name_or_path", model_name_or_path
                )

                print(f"Using PEFT with {peft_config.base_model_name_or_path}")

                if kwargs["args"].gradient_checkpointing:
                    self.model.enable_input_require_grads()
                self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model_kwargs = {
                "torch_dtype": self.model_dtype,
                "trust_remote_code": True,
            }
            if getattr(config, "model_type", None) == "gte":
                model_kwargs["unpad_inputs"] = False
                model_kwargs["use_memory_efficient_attention"] = False

            self.model = AutoModel.from_pretrained(
                model_name_or_path, config=config, **model_kwargs
            )

        self.pooling_strategy = pooling_strategy
        model_id = Path(model_name_or_path).name.lower()
        self.uses_embedding_head = "embeddinggemma" in model_id
        if self.uses_embedding_head:
            embedding_head = EmbeddingGemmaProjectionHead(model_name_or_path)
            dtype = getattr(self.model, "dtype", None)
            if dtype is not None:
                embedding_head = embedding_head.to(dtype=dtype)
            self.model.embedding_head = embedding_head
            self.pooling_strategy = "avg"
        else:
            self.model.embedding_head = None  # type: ignore[attr-defined]
        self.loss_fct = E5EmbeddingLoss(tau=0.02, matryoshka_dims=self.matryoshka_dims)

        super().__init__(model=self.model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )
        self.accelerator.print(self.model)
        if use_peft:
            # saving and loading checkpoints for resuming training
            self.accelerator.register_save_state_pre_hook(self.save_model_hook)
            self.accelerator.register_load_state_pre_hook(self.load_model_hook)

        self.use_homogeneous_batching = homogeneous_batching
        self.use_group_by_length = group_by_length

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        _, _ = model, num_items_in_batch
        _ = inputs.pop("labels", None)
        last_hidden_states = self.model(
            output_hidden_states=True, return_dict=True, **inputs
        ).hidden_states[-1]

        pooled_outputs = pool(last_hidden_states, inputs["attention_mask"], self.pooling_strategy)
        if getattr(self.model, "embedding_head", None) is not None:
            pooled_outputs = self.model.embedding_head(pooled_outputs)  # type: ignore[attr-defined]
        loss = self.loss_fct(pooled_outputs, self.num_sep_ids)

        return (loss, pooled_outputs) if return_outputs else loss

    def save_model_hook(self, models, weights, output_dir):
        for i, model in enumerate(models):
            model.save_pretrained(output_dir, state_dict=weights[i])
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(self, models, input_dir):
        while len(models) > 0:
            model = models.pop()
            # pop models so that they are not loaded again
            if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                model.load_adapter(input_dir, model.active_adapter, is_trainable=True)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        # Check for conflicting batching strategies
        if self.use_homogeneous_batching and self.use_group_by_length:
            raise ValueError(
                "[NotImplementedError] Cannot use both homogeneous_batching and group_by_length simultaneously. "
                "Please choose one batching strategy."
            )

        if self.use_group_by_length:
            dataloader_params = {
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }
            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["batch_sampler"] = GroupByLengthSampler(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    shuffle_batches=True,
                )
            return self.accelerator.prepare(
                DataLoader(train_dataset, **dataloader_params)  # type: ignore
            )

        if self.use_homogeneous_batching:
            dataloader_params = {
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }
            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["batch_sampler"] = DataHomogeneousSampler(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    fill_last_task=True,
                )
            return self.accelerator.prepare(
                DataLoader(train_dataset, **dataloader_params)  # type: ignore
            )
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
        return self.accelerator.prepare(
            DataLoader(train_dataset, **dataloader_params)  # type: ignore
        )

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override HF Trainer save to persist embedding head weights alongside checkpoints.
        """
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        if not self.accelerator.is_main_process:
            return

        resolved_output_dir = output_dir or self.args.output_dir
        if resolved_output_dir is None:
            return
        target_dir: str | Path = resolved_output_dir
        model_to_save = self.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module  # type: ignore[assignment]
        embedding_head = getattr(model_to_save, "embedding_head", None)
        if embedding_head is not None:
            save_embeddinggemma_modules(embedding_head, target_dir)
