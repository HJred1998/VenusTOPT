"""Batch samplers for protein residue embeddings."""
import math
import random
from typing import Iterator
from torch.utils.data import BatchSampler


class LengthBucketBatchSampler(BatchSampler):
    """Group similar-length proteins to reduce padding."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_size_multiplier: int = 20,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        buckets = [indices[i : i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        batches: list[list[int]] = []
        for bucket in buckets:
            bucket.sort(key=lambda i: self.lengths[i])
            local = [bucket[i : i + self.batch_size] for i in range(0, len(bucket), self.batch_size)]
            batches.extend(local)
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

class TemperatureStratifiedBatchSampler(BatchSampler):
    """Mix temperature ranges within training batches without oversampling."""

    def __init__(
        self,
        targets: list[float],
        batch_size: int,
        seed: int,
        bounds: tuple[float, float, float] = (37.0, 55.0, 70.0),
    ):
        self.targets = targets
        self.batch_size = batch_size
        self.seed = seed
        self.bounds = bounds
        self.epoch = 0
        self.groups: list[list[int]] = [[], [], [], []]
        b0, b1, b2 = bounds
        for idx, target in enumerate(targets):
            if target < b0:
                self.groups[0].append(idx)
            elif target < b1:
                self.groups[1].append(idx)
            elif target < b2:
                self.groups[2].append(idx)
            else:
                self.groups[3].append(idx)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [g.copy() for g in self.groups]
        for group in groups:
            rng.shuffle(group)
        nonempty = [i for i, group in enumerate(groups) if group]
        base_quota = max(1, self.batch_size // max(1, len(nonempty)))
        batches: list[list[int]] = []
        while any(groups):
            batch: list[int] = []
            for group_idx in nonempty:
                take = min(base_quota, len(groups[group_idx]), self.batch_size - len(batch))
                if take > 0:
                    batch.extend(groups[group_idx][-take:])
                    del groups[group_idx][-take:]
            while len(batch) < self.batch_size and any(groups):
                available = [i for i, group in enumerate(groups) if group]
                group_idx = rng.choice(available)
                batch.append(groups[group_idx].pop())
            if batch:
                rng.shuffle(batch)
                batches.append(batch)
        rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.targets) / self.batch_size)
