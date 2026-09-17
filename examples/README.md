# Example Prediction

`proteins.fasta` illustrates the accepted FASTA format. These artificial
sequences are provided to demonstrate the interface and have no experimental
Topt labels.

Run the following command from the repository root after obtaining the PRIME
model files:

```bash
python -m inference.predict \
  --input examples/proteins.fasta \
  --prime-model /path/to/PRIME \
  --output outputs/example_predictions.csv \
  --device cuda --precision fp32
```

The corresponding output is provided in
[`expected_predictions.csv`](expected_predictions.csv). It was generated with
the bundled VenusTOPT checkpoint and the PRIME files identified in
[`weights/PRIME_files.json`](../weights/PRIME_files.json), using FP32 inference.
Small numerical differences can occur across hardware and software versions;
reduced-precision inference can introduce larger differences.

| ID | Length | Predicted Topt (degrees Celsius) |
| --- | ---: | ---: |
| protein_1 | 22 | 40.412 |
| protein_2 | 47 | 43.701 |

The CSV contains `input_index` (zero-based input order), `id`, `sequence`,
`length`, and `predicted_topt_c`. These values demonstrate the prediction
interface and output format, not experimental accuracy or the biological
validity of these artificial sequences.

## CSV Input

For CSV input, use one record per row with an identifier and amino-acid sequence:

```csv
id,sequence
protein_1,MKTIIALSYIFCLVFADYKDDD
```

Predictions are written to the output path specified on the command line.
