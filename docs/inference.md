# Inference

## Input and Output

Use FASTA or UTF-8 CSV input with unique protein identifiers and nonempty
sequences containing the 20 canonical amino acids. CSV column names can be
specified with `--id-column` and `--sequence-column`.

Predictions are written in degrees Celsius. Existing output files are not
overwritten. A companion metadata file records input and model hashes and
execution settings. Predictions assist candidate prioritization; experimental
characterization remains necessary.

## PRIME Inference

```bash
python -m inference.predict --input proteins.fasta \
  --prime-model /path/to/PRIME --output outputs/predictions.csv --device cuda
```

PRIME files are loaded locally. Embeddings remain in memory. The supported
sequence interface is limited to 1,022 residues; longer inputs are rejected
without truncation. CPU execution is available with `--device cpu`.

FP32 is the default precision. On CUDA, `--precision fp16` or `--precision bf16`
enables reduced-precision PRIME computation. This can change predictions
slightly. The prediction head operates in FP32.

## Cached Embeddings

```bash
python -m inference.predict --input proteins.fasta \
  --embeddings-dir embeddings/prime --output outputs/predictions.csv
```

Files must be named `<id>.npy` with shape `[sequence_length,1280]`, excluding
BOS, EOS, and padding tokens. Use the matching PRIME model. Other PLMs are not
interchangeable merely because their dimensions match.

Explicit embedding export is available for training or repeated analysis:

```bash
python -m inference.extract_embeddings --input proteins.fasta \
  --prime-model /path/to/PRIME --output-dir embeddings/prime --device cuda
```

Only this command saves embeddings. The cached head can process longer arrays
subject to memory; an appropriate long-sequence embedding protocol must first
be established.

## Multi-GPU Inference

Launch one process per allocated GPU with the same input and shard count and
a unique shard ID. For a shell allocated two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python -m inference.predict --input proteins.fasta --prime-model /path/to/PRIME --output outputs/shard0.csv --shard-id 0 --num-shards 2 --device cuda &
CUDA_VISIBLE_DEVICES=1 python -m inference.predict --input proteins.fasta --prime-model /path/to/PRIME --output outputs/shard1.csv --shard-id 1 --num-shards 2 --device cuda &
wait
python -m inference.merge_shards --shards outputs/shard0.csv outputs/shard1.csv --output outputs/predictions.csv
```

On a cluster, follow the scheduler's GPU allocation. Merging checks that the
shards are complete and share input and model settings, then restores input
order. Partially completed shards are not resumed automatically.

## Numerical Considerations

The prediction interface processes proteins individually so results do not
depend on other input sequences. For direct batched calls, padding to a longer
sequence can change the set of overlapping terminal motif windows. Retain the
same batch construction when comparing batched evaluations. Numerical backends
and precision settings may also produce small differences.
