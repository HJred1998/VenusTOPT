"""Input validation and padded batches for residue-level protein representations."""
from dataclasses import dataclass
import csv
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from inference.common import embedding_path


@dataclass(frozen=True)
class Protein:
    identifier: str
    target: float
    path: Path
    length: int


def read_proteins(csv_path, embeddings_dir, input_dim):
    proteins, seen = [], set()
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'uniprot_id', 'topt'}.issubset(reader.fieldnames or []):
            raise ValueError(f'{csv_path}: columns uniprot_id and topt are required')
        for row in reader:
            uid = row['uniprot_id'].strip()
            if not uid or uid in seen:
                raise ValueError(f'{csv_path}: empty or duplicate ID {uid!r}')
            target = float(row['topt'])
            if not math.isfinite(target):
                raise ValueError(f'{uid}: target must be finite')
            path = embedding_path(embeddings_dir, uid)
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != input_dim:
                raise ValueError(f'{uid}: expected nonempty [L,{input_dim}] embedding')
            if not np.isfinite(array).all():
                raise ValueError(f'{uid}: embedding contains nonfinite values')
            proteins.append(Protein(uid, target, path, len(array)))
            seen.add(uid)
    if not proteins:
        raise ValueError(f'{csv_path}: no records')
    return proteins


def require_disjoint(*splits):
    seen = set()
    for split in splits:
        identifiers = {p.identifier for p in split}
        overlap = seen & identifiers
        if overlap:
            raise ValueError(f'Overlapping split IDs: {sorted(overlap)[:5]}')
        seen.update(identifiers)


class ProteinDataset(Dataset):
    def __init__(self, proteins, target_mean, target_std):
        self.proteins = proteins
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        protein = self.proteins[index]
        values = np.load(protein.path, allow_pickle=False).astype(np.float32)
        return (protein.identifier, torch.from_numpy(values),
                (protein.target - self.target_mean) / self.target_std, protein.target)


def collate_proteins(batch):
    max_length = max(item[1].size(0) for item in batch)
    features = torch.zeros(len(batch), max_length, batch[0][1].size(1))
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for index, (_, values, _, _) in enumerate(batch):
        features[index, :len(values)] = values
        mask[index, :len(values)] = True
    return {
        'ids': [item[0] for item in batch], 'features': features, 'mask': mask,
        'target_normalized': torch.tensor([item[2] for item in batch], dtype=torch.float32),
        'target': torch.tensor([item[3] for item in batch], dtype=torch.float32),
    }
