from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')


def read_sequences(path, id_column='id', sequence_column='sequence'):
    path = Path(path)
    if path.suffix.lower() in ('.fa', '.faa', '.fasta'):
        records = []
        uid, parts = None, []
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if uid is not None:
                    records.append((uid, ''.join(parts)))
                header = line[1:].split()
                if not header:
                    raise ValueError('FASTA header must contain a protein identifier')
                uid, parts = header[0], []
            elif uid is None:
                raise ValueError('FASTA sequence found before first header')
            else:
                parts.append(line)
        if uid is not None:
            records.append((uid, ''.join(parts)))
    else:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {id_column, sequence_column}.issubset(reader.fieldnames or []):
                raise ValueError(f'CSV requires {id_column!r} and {sequence_column!r}')
            records = [(r[id_column], r[sequence_column]) for r in reader]
    if not records:
        raise ValueError('No input sequences')
    cleaned, seen = [], set()
    for uid, sequence in records:
        uid = uid.strip()
        sequence = ''.join(sequence.split()).upper()
        if not uid or uid in seen:
            raise ValueError(f'Empty or duplicate sequence ID: {uid!r}')
        if not sequence or set(sequence) - AMINO_ACIDS:
            raise ValueError(f'{uid}: expected a nonempty sequence of 20 canonical amino acids')
        cleaned.append((uid, sequence))
        seen.add(uid)
    return cleaned


def embedding_path(directory, uid):
    if uid in ('.', '..') or any(c in uid for c in '/\\:'):
        raise ValueError(f'ID cannot be used as an embedding filename: {uid!r}')
    return Path(directory) / (uid + '.npy')


def load_embedding(directory, uid, length, dim):
    x = np.load(embedding_path(directory, uid), allow_pickle=False).astype(np.float32)
    if x.shape != (length, dim) or not np.isfinite(x).all():
        raise ValueError(f'{uid}: expected finite embedding of shape {(length, dim)}, got {x.shape}')
    return x


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
