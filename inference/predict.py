"""Predict from FASTA/CSV and local PRIME weights or existing residue embeddings."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from models import load_predictor
from .common import read_sequences, load_embedding, write_csv
from .prime import PrimeEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prime-model', type=Path)
    source.add_argument('--embeddings-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/VenusTOPT.pt'))
    parser.add_argument('--config', type=Path, default=Path('configs/model.json'))
    parser.add_argument('--id-column', default='id')
    parser.add_argument('--sequence-column', default='sequence')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error('Require 0 <= shard-id < num-shards')
    if args.output.exists():
        parser.error('Output already exists; choose a new output path')
    records = read_sequences(args.input, args.id_column, args.sequence_column)
    selected = [(i, r) for i, r in enumerate(records) if i % args.num_shards == args.shard_id]
    if not selected:
        parser.error('This shard has no records')
    model, config = load_predictor(args.checkpoint, args.config, args.device)
    encoder = PrimeEncoder(args.prime_model, args.device, args.precision) if args.prime_model else None
    if encoder and any(len(sequence) > encoder.max_residues for _, (_, sequence) in selected):
        parser.error(f'Input exceeds supported PRIME length ({encoder.max_residues}); no truncation was applied')
    rows = []
    with torch.inference_mode():
        for number, (index, (uid, sequence)) in enumerate(selected, 1):
            x = encoder.encode(sequence) if encoder else torch.from_numpy(
                load_embedding(args.embeddings_dir, uid, len(sequence), config['input_dim'])
            ).unsqueeze(0).to(args.device)
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=args.device)
            prediction = (model(x, mask) * config['target_std'] + config['target_mean']).item()
            if not torch.isfinite(torch.tensor(prediction)):
                raise ValueError(f'{uid}: nonfinite prediction')
            rows.append({'input_index': index, 'id': uid, 'sequence': sequence,
                         'length': len(sequence), 'predicted_topt_c': prediction})
            print(f'[{number}/{len(selected)}] {uid}: {prediction:.3f} C', flush=True)
    write_csv(args.output, rows)
    args.output.with_suffix('.metadata.json').write_text(json.dumps({
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'shard_id': args.shard_id, 'num_shards': args.num_shards,
        'input_records': len(records), 'output_records': len(rows),
        'device': args.device, 'precision': args.precision, 'head_batch_size': 1,
        'torch_version': torch.__version__,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
