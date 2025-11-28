import random

import numpy as np
from torch.utils.data import BatchSampler


class GroupByLengthSampler(BatchSampler):
    """
    Batch sampler that groups samples by average document length.
    Batches with shorter documents will be created first (curriculum learning).

    Each sample contains multiple documents (query + positive + negatives),
    and we calculate the average length across all documents in a sample.
    """

    def __init__(self, my_dataset, batch_size, drop_last, shuffle_batches=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_batches = shuffle_batches

        # Calculate average document length for each sample
        sample_lengths = []
        for idx in range(len(my_dataset)):
            # seperate_ids marks which document each token belongs to
            # e.g., [0,0,0,1,1,1,2,2,2] means 3 documents with 3 tokens each
            seperate_ids = my_dataset[idx]["seperate_ids"]

            # Count tokens for each document
            unique_ids = set(seperate_ids)
            doc_lengths = [seperate_ids.count(doc_id) for doc_id in unique_ids]

            # Average length across all documents (query + positive + negatives)
            avg_length = sum(doc_lengths) / len(doc_lengths)
            sample_lengths.append((idx, avg_length))

        # Sort by average length (ascending - shorter samples first)
        sample_lengths.sort(key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in sample_lengths]

        # Create batches sequentially from sorted indices
        self.batches = []
        for i in range(0, len(sorted_indices), self.batch_size):
            batch = sorted_indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                self.batches.append(batch)

        # Optionally shuffle batch order (but keep samples within each batch together)
        if self.shuffle_batches:
            random.shuffle(self.batches)

    def __iter__(self):
        batch_iter = iter(self.batches)
        for _ in range(len(self)):
            yield next(batch_iter)

    def __len__(self):
        return len(self.batches)


class DataHomogeneousSampler(BatchSampler):
    def __init__(self, my_dataset, batch_size, drop_last, fill_last_task):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.fill_last_task = fill_last_task

        self.dataset_types = np.array(my_dataset["task_type"])
        self.dataset_indices = {}
        for idx, t in enumerate(self.dataset_types):
            if t in self.dataset_indices:
                self.dataset_indices[t].append(idx)
            else:
                self.dataset_indices[t] = [idx]

        # random sampling을 위해서 indices를 섞기
        for k in self.dataset_indices:
            random.shuffle(self.dataset_indices[k])

        # merged indices가 해당 sampler의 최종 sampling indices sequence가 될 것임
        self.merged_indices = []
        for k in self.dataset_indices:
            if self.fill_last_task:
                remainder = len(self.dataset_indices[k]) % self.batch_size
                if remainder > 0:
                    self.dataset_indices[k].extend(
                        random.sample(
                            self.dataset_indices[k][:-remainder],
                            self.batch_size - remainder,
                        )
                    )
            self.merged_indices.extend(self.dataset_indices[k])

        # create batch
        self.batches = []
        merged_indices_iter = iter(self.merged_indices)
        for _ in range(len(self)):
            batch = []
            for _ in range(self.batch_size):
                batch.append(next(merged_indices_iter))
            # check all indices in batch are unique
            assert len(set(batch)) == len(batch)
            # check all task_type in batch are same
            assert len(set(self.dataset_types[batch])) == 1
            self.batches.append(batch)
        random.shuffle(self.batches)

    def __iter__(self):
        batch_iter = iter(self.batches)
        for _ in range(len(self)):
            yield next(batch_iter)

    def __len__(self):
        if self.drop_last:
            return len(self.merged_indices) // self.batch_size
        return (len(self.merged_indices) + self.batch_size - 1) // self.batch_size
