"""Merge complete prediction shards, validating input and model provenance."""
import argparse
import csv
import json
from pathlib import Path
from .common import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    all_rows, metas = [], []
    for path in args.shards:
        metas.append(json.loads(path.with_suffix('.metadata.json').read_text()))
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        meta = metas[-1]
        expected = set(range(meta['shard_id'], meta['input_records'], meta['num_shards']))
        observed = [int(r['input_index']) for r in rows]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(f'Incomplete/duplicate shard: {path}')
        all_rows.extend(rows)
    for key in ('input_sha256', 'checkpoint_sha256', 'config_sha256', 'num_shards', 'input_records', 'precision'):
        if len({m[key] for m in metas}) != 1:
            raise ValueError(f'Shard metadata differ: {key}')
    count = metas[0]['num_shards']
    if len(metas) != count or {m['shard_id'] for m in metas} != set(range(count)):
        raise ValueError('All shards must be supplied exactly once')
    all_rows.sort(key=lambda r: int(r['input_index']))
    write_csv(args.output, all_rows)
    print(f'Merged {len(all_rows)} predictions into {args.output}')


if __name__ == '__main__':
    main()
