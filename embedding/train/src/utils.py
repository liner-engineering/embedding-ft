import math

import torch.distributed as dist
from datasets import concatenate_datasets, load_dataset
from omegaconf import OmegaConf


def is_main_process():
    """Check if current process is the main process (rank 0)"""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_instruct_template(task_type: str) -> str:
    task_type = task_type.strip().lower()
    if task_type == "bookmrc":
        task_description = TaskDescription.BookMRC
    elif task_type == "mrc":
        task_description = TaskDescription.MRC
    elif task_type == "commonsense":
        task_description = TaskDescription.CommonSense
    elif task_type == "korquad1":
        task_description = TaskDescription.KorQuad1
    elif task_type == "korquad2":
        task_description = TaskDescription.KorQuad2
    elif task_type == "msmarco_en":
        task_description = TaskDescription.MSMARCO_EN
    else:
        raise ValueError(f"task_type {task_type} not supported")
    return f"Instruct: {task_description}\nQuery: {{text}}"


class TaskDescription:
    """
    Predefined task descriptions. Follow the model usage to choose the corresponding task description.

    Example::

            import TaskDescription

            # list all pre-defined task descriptions
            print(TaskDescription.list_task_descriptions())
            # get task description
            print(TaskDescription.Book_MRC)

    """

    BookMRC = "Given a question, retrieve passages that answer the question"
    MRC = "Given a question, retrieve articles that answer the question"
    CommonSense = "Given a question, retrieve Wikipedia passages that answer the question"
    KorQuad1 = "Given a question, retrieve Wikipedia passages that answer the question"
    KorQuad2 = "Given a question, retrieve Wikipedia passages that answer the question"
    MSMARCO_EN = "Given a web search query, retrieve relevant passages that answer the query"

    @classmethod
    def list_task_descriptions(cls):
        for key, val in TaskDescription.__dict__.items():
            if key.startswith("_") or key == "list_task_descriptions" or key == "list_task_types":
                continue
            print(f"TaskDescription.{key}", "=", f"'{val}'")

    @classmethod
    def list_task_types(cls):
        keys = []
        for key, val in TaskDescription.__dict__.items():
            if key.startswith("_") or key == "list_task_descriptions" or key == "list_task_types":
                continue
            keys.append(key)
        return keys


def load_dataset_from_config(cfg):
    # set_seed(42)
    cfg = OmegaConf.load(cfg)
    dataset = []
    if cfg.use_all:
        data_files = cfg.datasets["data_path"]
        data_names = cfg.datasets["name"]
        for name, data_file in zip(data_names, data_files):
            data = load_dataset("json", data_files=data_file)
            data = data["train"].shuffle()  # type: ignore

            # Deduplicate data based on text + positive combination
            original_len = len(data)
            seen_keys = set()
            dedup_indices = []

            for idx in range(len(data)):
                text = data[idx]["text"].lower().strip()
                positive = data[idx]["positive"].lower().strip()
                key = f"{text}||{positive}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    dedup_indices.append(idx)

            data = data.select(dedup_indices)
            deduped_count = original_len - len(data)

            if is_main_process():
                print(
                    f"[{name}] Removed {deduped_count} duplicate entries (original: {original_len}, after dedup: {len(data)})"
                )
                print(f"Sample data: {data[0]}")

            dataset.append({"data": data, "num_data": len(data), "task_type": name})
        return dataset

    num_total_data = cfg.num_total_data
    data_ratio = cfg.datasets["data_ratio"]
    data_names = cfg.datasets["name"]
    data_files = cfg.datasets["data_path"]

    # set data ratio to sum 1
    ratio_sum = sum(data_ratio)
    data_ratio = [ratio / ratio_sum for ratio in data_ratio]

    for name, ratio, data_file in zip(data_names, data_ratio, data_files):
        data = load_dataset("json", data_files=data_file)  # type: ignore
        data = data["train"].shuffle()  # type: ignore

        # dataset num assigned by given ratio
        assigned_num_data = math.floor(num_total_data * ratio)

        # if assigned_num_data is less than the number of data, select assigned_num_data data
        if assigned_num_data <= len(data):
            data = data.select(range(assigned_num_data))
        else:
            # if assigned_num_data is more than the number of data
            num_data = len(data)
            if cfg.allow_duplicate:
                data1 = data.select(range(num_data))
                assigned_num_data -= num_data
                while assigned_num_data > 0:
                    if assigned_num_data >= num_data:
                        data1 = concatenate_datasets([data1, data])
                        assigned_num_data -= num_data
                    else:
                        data2 = data.select(range(assigned_num_data))
                        data1 = concatenate_datasets([data1, data2])
                        assigned_num_data -= num_data
                data = data1

        # Deduplicate data based on text + positive combination
        original_len = len(data)
        seen_keys = set()
        dedup_indices = []

        for idx in range(len(data)):
            text = data[idx]["text"].lower().strip()
            positive = data[idx]["positive"].lower().strip()
            key = f"{text}||{positive}"
            if key not in seen_keys:
                seen_keys.add(key)
                dedup_indices.append(idx)

        data = data.select(dedup_indices)
        deduped_count = original_len - len(data)

        if is_main_process():
            print(
                f"[{name}] Removed {deduped_count} duplicate entries (original: {original_len}, after dedup: {len(data)})"
            )

        dataset.append({"data": data, "num_data": len(data), "task_type": name})

    return dataset
