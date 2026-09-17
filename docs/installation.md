# Installation

Use **Python 3.10** in a new environment. Install dependencies from the
repository root. The pinned requirements cover prediction, embedding export,
training and evaluation.

## Create an Environment

Using Conda on Linux or Windows:

```bash
conda create -n venustopt python=3.10 -y
conda activate venustopt
python -m pip install --upgrade pip
```

A standard Python virtual environment is also suitable. Avoid installing into
an environment that already contains unrelated machine-learning packages.

## Install PyTorch

Choose **one** of the following builds. Use an up-to-date NVIDIA driver for
GPU execution; installing Python dependencies does not install the GPU driver.

For an NVIDIA GPU with a driver compatible with CUDA 12.6:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For CPU execution:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

These commands follow the [PyTorch version-specific installation instructions](https://pytorch.org/get-started/previous-versions/#v271).
Other supported PyTorch 2.7.1 builds can be selected there for a compatible
platform. Do not infer GPU support from the presence of the `torch` package alone.

## Install VenusTOPT Dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The `torch==2.7.1` requirement accepts its corresponding CPU or CUDA build,
so installing the requirements preserves a matching selected build.
Install the requirements from the normal Python package index, not from a
PyTorch-only index.

| Package | Role |
| --- | --- |
| PyTorch | Model execution and training |
| Transformers | PRIME model and tokenizer loading |
| Tokenizers, Hugging Face Hub | Transformers runtime dependencies |
| Safetensors | PRIME weight loading |
| NumPy | Residue embeddings and numerical operations |
| SciPy | Evaluation correlations |

Additional transitive dependencies are installed automatically by pip.
VenusTOPT does not require torchvision, torchaudio, pandas, Biopython, PyMOL,
or the external flash-attn package. The compatible PRIME implementation uses
PyTorch's scaled dot-product attention.

## Model Files

Python dependencies are separate from model files. Sequence-to-temperature
inference also requires the bundled `weights/VenusTOPT.pt`, its matching
`configs/model.json`, and a compatible local PRIME model directory. Obtain PRIME
from its [official repository](https://github.com/ai4protein/Pro-Prime) following
the instructions and terms there. PRIME weights are not bundled or downloaded
automatically. See [pretrained weights](../weights/README.md) for the complete
file list. The dependencies above cover PRIME loading within VenusTOPT; the
upstream project's full environment is not required for this workflow.

```bash
python -m inference.predict \
  --input examples/proteins.fasta \
  --prime-model /path/to/PRIME \
  --output outputs/predictions.csv \
  --device cuda
```

For CPU execution, replace `--device cuda` with `--device cpu`.

## Troubleshooting

- **CUDA unavailable:** verify the installed PyTorch build, driver, and assigned
  GPU. CPU execution can be selected explicitly.
- **Package version conflicts:** recreate a clean Python 3.10 environment and
  install the pinned requirements. Use `python -m pip` so installation targets
  the interpreter that runs the model.
- **Build-dependency or wheel metadata errors:** upgrade pip before installing
  PyTorch. An outdated resolver can misinterpret dependency metadata.
- **Missing PRIME files:** provide the full model directory, including its
  custom Python files and tokenizer resources. Requirements installation does
  not download this backbone.
