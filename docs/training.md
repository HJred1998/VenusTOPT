# Training

## Inputs

Supply a training CSV and a directory of PRIME residue embeddings. Each CSV
requires `uniprot_id` and `topt` columns. Targets are in degrees Celsius. Each
ID must correspond to a finite `<uniprot_id>.npy` array of shape `[L,1280]`,
excluding special and padding tokens. IDs must be unique and splits disjoint.

```bash
python -m training.train \
  --train-csv data/private/train.csv \
  --val-csv data/private/validation.csv \
  --test-csv data/private/test.csv \
  --embeddings-dir embeddings/prime \
  --output-dir outputs/training
```

Without `--val-csv`, 10% of the training records form an internal validation
set. The split uses `validation_seed` in `configs/training.json`; input CSV order
must be kept fixed for reproducibility. The `seed` setting controls parameter
initialization and training randomness; `--seed` overrides this setting.
Sequence-similarity splits must be
prepared before training; the training command does not cluster sequences.

The script reads embeddings from disk, pads within each batch, and masks padded
positions. It validates IDs, targets, dimensions, and embedding finiteness before
training. ES-Topt data and split assignments are not bundled.

## Optimization

| Setting | Default |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0003 |
| Batch size | 48 |
| Loss | Smooth L1, beta=0.2 |
| Gradient clipping norm | 1.0 |
| Maximum epochs | 250 |
| Early-stopping patience | 45 epochs |
| Checkpoint selection | Validation MAE, starting at epoch15 |

Targets are standardized using the training-fit subset only. The loss parameter
beta is in standardized-target units. The training sampler mixes temperature
ranges within batches without oversampling. Validation batches are grouped by
sequence length. Deterministic PyTorch algorithms are enabled.

To customize optimization, provide a JSON configuration through
`--training-config`. The model architecture is specified by `--model-config`.
For CPU training use `--device cpu`; `--num-threads` controls PyTorch CPU threads.

## Outputs

- `VenusTOPT.pt`: model selected by validation MAE
- `model.json`: architecture and fitted target normalization
- `training.json`: optimization settings and random seed
- `training_history.csv`: epoch-level loss and validation MAE
- `split_membership.csv`: record IDs and their split assignments
- `validation_predictions.csv` and optional `test_predictions.csv`
- `metrics.json`: selected epoch and evaluation metrics

The test split is optional and does not influence checkpoint selection. Training
results can vary with random initialization, embeddings, and numerical backend.
Use the provided pretrained model for inference without retraining.

Load a newly trained model with:

```bash
python -m inference.predict --input proteins.fasta \
  --prime-model /path/to/PRIME --output outputs/predictions.csv \
  --checkpoint outputs/training/VenusTOPT.pt \
  --config outputs/training/model.json
```
