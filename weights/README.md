# Pretrained Weights

## VenusTOPT

`VenusTOPT.pt` contains the pretrained prediction-head state dictionary.
Load it with the accompanying `configs/model.json`, which defines the model
architecture and target normalization. File integrity can be checked using
`checksums.sha256`.

The bundled VenusTOPT prediction-head weights are covered by the repository's
[MIT License](../LICENSE). This does not extend to the separately distributed
PRIME model.

## PRIME

PRIME is the separate, frozen protein language model used to encode sequences.
Obtain the pretrained PRIME model from the
[official PRIME repository](https://github.com/ai4protein/Pro-Prime), following
the instructions and terms provided there. VenusTOPT does not redistribute
PRIME weights or maintain the upstream PRIME project.

Download the complete model directory, not only the weight file. The directory
passed to `--prime-model` must contain:

```text
PRIME/
|-- config.json
|-- model.safetensors
|-- configuration_proprime.py
|-- modeling_proprime.py
|-- tokenization_proprime.py
|-- tokenizer_config.json
|-- special_tokens_map.json
`-- vocab.txt
```

`PRIME_files.json` identifies the expected files by size and SHA-256 hash.
Use these file identifiers to check compatibility with the released VenusTOPT
checkpoint.

From the VenusTOPT repository root, run:

```bash
python -m inference.predict \
  --input examples/proteins.fasta \
  --prime-model /path/to/PRIME \
  --output outputs/predictions.csv \
  --device cuda
```

The VenusTOPT inference script loads PRIME and computes residue embeddings in
memory before predicting Topt. Running PRIME notebooks or extracting embeddings
separately is not required. For this workflow, use the VenusTOPT
[installation instructions](../docs/installation.md); installing the upstream
PRIME project's full requirements is not necessary.

The loader operates offline with `local_files_only=True`. PRIME uses custom
Python model/tokenizer files loaded through `trust_remote_code=True`; use a
trusted model directory.
