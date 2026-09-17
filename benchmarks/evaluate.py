"""Calculate regression metrics without silently dropping missing observations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) < 2 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Require at least two finite experimental/predicted pairs')
    errors = p - y
    varying = np.ptp(y) > 0 and np.ptp(p) > 0
    return {'n': len(y), 'mae': float(np.abs(errors).mean()),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'r2': float(1 - np.sum(errors ** 2) / np.sum((y - y.mean()) ** 2)) if np.ptp(y) > 0 else None,
            'pearson': float(pearsonr(y, p).statistic) if varying else None,
            'spearman': float(spearmanr(y, p).statistic) if varying else None,
            'bias': float(errors.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--target-column', default='topt')
    parser.add_argument('--prediction-column', default='predicted_topt_c')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    result = metrics([r[args.target_column] for r in rows], [r[args.prediction_column] for r in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
