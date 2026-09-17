"""Train VenusTOPT on user-supplied protein splits and frozen PRIME embeddings."""
import argparse
import csv
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import VenusTOPT
from benchmarks.evaluate import metrics
from inference.common import write_csv
from .data import ProteinDataset, collate_proteins, read_proteins, require_disjoint
from .samplers import LengthBucketBatchSampler, TemperatureStratifiedBatchSampler


def set_random_seed(seed, deterministic=True):
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True)


def make_loader(proteins, model_config, settings, seed, training=False):
    dataset = ProteinDataset(proteins, model_config['target_mean'], model_config['target_std'])
    if training:
        sampler = TemperatureStratifiedBatchSampler(
            [p.target for p in proteins], settings['batch_size'], seed,
            tuple(settings['sampler_temperature_bounds']),
        )
    else:
        sampler = LengthBucketBatchSampler(
            [p.length for p in proteins], settings['batch_size'], shuffle=False, seed=seed,
        )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_proteins,
                      num_workers=0, pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def evaluate(model, loader, config, device):
    model.eval()
    rows, error_sum, count = [], 0.0, 0
    for batch in loader:
        prediction = model(batch['features'].to(device), batch['mask'].to(device))
        normalized_target = batch['target_normalized'].to(device)
        error_sum += float((prediction - normalized_target).abs().sum().item())
        count += len(prediction)
        prediction = prediction.cpu() * config['target_std'] + config['target_mean']
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite evaluation predictions')
        rows.extend({'uniprot_id': uid, 'topt': float(target), 'predicted_topt_c': float(pred)}
                    for uid, target, pred in zip(batch['ids'], batch['target'], prediction))
    return error_sum / count, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path)
    parser.add_argument('--test-csv', type=Path)
    parser.add_argument('--embeddings-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, default=Path('configs/model.json'))
    parser.add_argument('--training-config', type=Path, default=Path('configs/training.json'))
    parser.add_argument('--seed', type=int, help='Override the random seed in the training configuration')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-threads', type=int, default=4)
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error('--num-threads must be positive')
    torch.set_num_threads(args.num_threads)
    config = json.loads(args.model_config.read_text(encoding='utf-8'))
    settings = json.loads(args.training_config.read_text(encoding='utf-8'))
    if args.seed is None:
        args.seed = settings['seed']
    for key in ('batch_size', 'epochs', 'checkpoint_start_epoch', 'early_stopping_patience'):
        if settings[key] < 1:
            parser.error(f'{key} must be positive')
    if settings['checkpoint_start_epoch'] > settings['epochs']:
        parser.error('checkpoint_start_epoch must not exceed epochs')
    for key in ('learning_rate', 'smooth_l1_beta', 'gradient_clip_norm'):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            parser.error(f'{key} must be positive and finite')
    if not 0 < settings['validation_fraction'] < 1:
        parser.error('validation_fraction must be between zero and one')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose an empty training output directory')

    train = read_proteins(args.train_csv, args.embeddings_dir, config['input_dim'])
    if args.val_csv:
        validation = read_proteins(args.val_csv, args.embeddings_dir, config['input_dim'])
    else:
        indices = list(range(len(train)))
        random.Random(settings['validation_seed']).shuffle(indices)
        n_val = max(1, int(round(len(train) * settings['validation_fraction'])))
        if n_val >= len(train):
            parser.error('Too few training proteins for an internal validation split')
        validation = [train[i] for i in indices[:n_val]]
        train = [train[i] for i in indices[n_val:]]
    test = read_proteins(args.test_csv, args.embeddings_dir, config['input_dim']) if args.test_csv else []
    require_disjoint(train, validation, test)
    targets = np.asarray([p.target for p in train], dtype=np.float32)
    config['target_mean'] = float(targets.mean())
    config['target_std'] = float(targets.std())
    if config['target_std'] < 1e-6:
        config['target_std'] = 1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'model.json').write_text(json.dumps(config, indent=2) + '\n')
    (args.output_dir / 'training.json').write_text(json.dumps({**settings, 'seed': args.seed}, indent=2) + '\n')
    write_csv(args.output_dir / 'split_membership.csv', [
        {'uniprot_id': p.identifier, 'split': split_name}
        for split_name, proteins in (('train', train), ('validation', validation), ('test', test))
        for p in proteins
    ])

    set_random_seed(args.seed)
    train_loader = make_loader(train, config, settings, args.seed, training=True)
    val_loader = make_loader(validation, config, settings, args.seed)
    model = VenusTOPT(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    best_score, best_epoch = float('inf'), 0
    patience = settings['early_stopping_patience']
    checkpoint_path = args.output_dir / 'VenusTOPT.pt'
    with (args.output_dir / 'training_history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'training_loss', 'validation_mae_c'])
        writer.writeheader()
        for epoch in range(1, settings['epochs'] + 1):
            model.train()
            train_loader.batch_sampler.epoch = epoch
            losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch['features'].to(args.device), batch['mask'].to(args.device))
                target = batch['target_normalized'].to(args.device)
                per_sample = F.smooth_l1_loss(prediction, target, beta=settings['smooth_l1_beta'], reduction='none')
                loss = per_sample.sum() / per_sample.numel()
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite training loss at epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            score, _ = evaluate(model, val_loader, config, args.device)
            if epoch >= settings['checkpoint_start_epoch']:
                if score < best_score - 1e-6:
                    best_score, best_epoch = score, epoch
                    patience = settings['early_stopping_patience']
                    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, checkpoint_path)
                else:
                    patience -= 1
            row = {'epoch': epoch, 'training_loss': float(np.mean(losses)),
                   'validation_mae_c': score * config['target_std']}
            writer.writerow(row)
            handle.flush()
            print(f"Epoch {epoch}: loss={row['training_loss']:.5f}, validation MAE={row['validation_mae_c']:.3f} C", flush=True)
            if patience <= 0:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    summary = {'best_epoch': best_epoch, 'validation_mae_c': best_score * config['target_std']}
    for split_name, proteins in (('validation', validation), ('test', test)):
        if not proteins:
            continue
        _, rows = evaluate(model, make_loader(proteins, config, settings, args.seed), config, args.device)
        write_csv(args.output_dir / f'{split_name}_predictions.csv', rows)
        if len(rows) >= 2:
            summary[split_name] = metrics([r['topt'] for r in rows], [r['predicted_topt_c'] for r in rows])
    (args.output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'Saved {checkpoint_path}', flush=True)


if __name__ == '__main__':
    main()
