"""Optional explicit embedding export for training or repeated evaluation."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .prime import PrimeEncoder
from .common import read_sequences, embedding_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--prime-model', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--id-column', default='id')
    parser.add_argument('--sequence-column', default='sequence')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    records = read_sequences(args.input, args.id_column, args.sequence_column)
    paths = [embedding_path(args.output_dir, uid) for uid, _ in records]
    if any(p.exists() for p in paths):
        parser.error('Output embeddings already exist; use a fresh directory')
    encoder = PrimeEncoder(args.prime_model, args.device)
    if any(len(seq) > encoder.max_residues for _, seq in records):
        parser.error(f'Sequence longer than {encoder.max_residues}; no truncation was applied')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (uid, seq), path in zip(records, paths):
        np.save(path, encoder.encode(seq)[0].cpu().numpy(), allow_pickle=False)
        print(uid, flush=True)


if __name__ == '__main__':
    main()
