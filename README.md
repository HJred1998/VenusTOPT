# VenusTOPT

**Sequence-only prediction of enzyme optimal catalytic temperature**

VenusTOPT predicts enzyme optimal catalytic temperature (`Topt`) directly from
protein sequence. It combines temperature-informed PRIME representations with
multi-scale local motif modeling and global protein feature decoding.

The associated study introduces **ES-Topt**, an evidence-supported dataset
curated from primary literature, and demonstrates temperature-guided
endoglucanase mining with experimental validation.

The manuscript is currently under review. **The dataset is not included in this
repository at this stage.** Dataset availability will be updated separately.

## Highlights

- Predicts `Topt` without protein structures or organism growth temperatures.
- Integrates local sequence motifs with global protein representations.
- Supports FASTA and CSV inputs, cached embeddings, and multi-GPU inference.
- Provides pretrained weights, training code, and regression evaluation tools.

## Installation

Use a new **Python 3.10** environment. For an NVIDIA GPU with a CUDA
12.6-compatible driver, run the following from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

A CUDA GPU is recommended for PRIME inference. CPU installation and environment
setup are described in the [installation guide](docs/installation.md).

## Pretrained Model

The VenusTOPT prediction head is provided in `weights/VenusTOPT.pt`, with its
configuration in `configs/model.json`.

Download the pretrained PRIME model from the
[official PRIME repository](https://github.com/ai4protein/Pro-Prime), following
its model-access instructions and terms. PRIME weights are not redistributed
in this repository. Keep the complete model directory, including its tokenizer
and custom Python files, and supply its path with `--prime-model`.

VenusTOPT handles PRIME encoding and temperature prediction in one command;
there is no need to run the PRIME notebooks or save embeddings first. Install
the dependencies provided here for this workflow. See
[model weights](weights/README.md) for the required files.

## Quick Start

Run all commands from the repository root.

```bash
python -m inference.predict \
  --input examples/proteins.fasta \
  --prime-model /path/to/PRIME \
  --output outputs/predictions.csv \
  --device cuda
```

The output CSV contains `id`, `sequence`, `length`, and `predicted_topt_c`
(degrees Celsius), together with the original row index. Embeddings are computed
in memory and are not saved during prediction.

For CSV input, the default columns are `id` and `sequence`:

```bash
python -m inference.predict \
  --input proteins.csv \
  --id-column protein_id --sequence-column sequence \
  --prime-model /path/to/PRIME \
  --output outputs/predictions.csv
```

The sequence interface accepts the 20 canonical amino acids and sequences up to
1,022 residues. Longer sequences are rejected without truncation. For CPU
inference, use `--device cpu`. See [inference documentation](docs/inference.md)
for precision options, cached embeddings, and multi-GPU execution.

## Model

PRIME residue embeddings are projected to a shared feature space. The global
module combines query attention pooling with mean, max, and standard-deviation
pooling. The local module forms attentive motif tokens from overlapping windows
at multiple scales. Cross-attention uses the global query features to aggregate
local motif information. Their representations are concatenated and passed to
a regression head.

See [model architecture](docs/architecture.md) for the data flow and dimensions.

## Training

Prepare protein-level splits and matching PRIME residue embeddings, then run:

```bash
python -m training.train \
  --train-csv data/private/train.csv \
  --val-csv data/private/validation.csv \
  --test-csv data/private/test.csv \
  --embeddings-dir embeddings/prime \
  --output-dir outputs/training
```

VenusTOPT is trained with AdamW and Smooth L1 loss on standardized `Topt` targets.
Settings are defined in `configs/training.json`. The test split is optional and
is evaluated only after training. See [training documentation](docs/training.md)
for input formats and checkpoint selection.

## Evaluation

Evaluate experimental and predicted temperatures in a CSV:

```bash
python -m benchmarks.evaluate \
  --csv labeled_predictions.csv \
  --target-column topt --prediction-column predicted_topt_c \
  --output outputs/metrics.json
```

The script reports MAE, RMSE, R2, Pearson correlation, Spearman correlation, and
prediction bias.

## Repository Structure

```text
VenusTOPT/
|-- models/            # VenusTOPT architecture
|-- weights/           # Pretrained prediction head
|-- configs/           # Model and training settings
|-- inference/         # Prediction and embedding utilities
|-- training/          # Model training
|-- benchmarks/        # Regression evaluation
|-- examples/          # Example input format
|-- data/              # Data availability and input schema
|-- docs/              # Usage and model documentation
`-- requirements.txt
```

## Related Work

**PRIME** and **VenusMine** were developed in previous studies by our group.
[PRIME](https://github.com/ai4protein/Pro-Prime) provides the temperature-informed
representations used by VenusTOPT. VenusMine is the enzyme-mining framework into
which VenusTOPT was integrated for experimental validation.

- **PRIME:** Jiang, F. et al. A general temperature-guided language model to
  design proteins of enhanced stability and activity. *Science Advances* **10**,
  eadr2641 (2024).
- **VenusMine:** Wu, B. et al. Harnessing protein language model for
  structure-based discovery of highly efficient and robust PET hydrolases.
  *Nature Communications* **16**, 6211 (2025).

## Citation

If you use VenusTOPT, please cite the associated manuscript. Citation information
will be updated upon publication.

## License

The VenusTOPT source code and bundled prediction-head weights are released
under the [MIT License](LICENSE). PRIME and other third-party dependencies
remain subject to their respective licenses and terms; this license does not
relicense those components. The ES-Topt dataset is not included in this release.

## Contact

For questions about installation or use, please open a GitHub issue.
For research inquiries, contact Jin Huang at jinhuang.sjtu@gmail.com.
