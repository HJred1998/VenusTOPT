# VenusTOPT

**Sequence-only prediction of enzyme optimal catalytic temperature**

## Overview

The optimal catalytic temperature (`Topt`) of an enzyme is an important
consideration when matching biocatalysts to the thermal conditions of
industrial and laboratory processes. However, experimental determination of
`Topt` is labor-intensive and difficult to perform at the scale required for
large enzyme-mining campaigns.

VenusTOPT is a sequence-only deep learning model designed for scalable
prediction of enzyme `Topt` directly from protein sequences. The model is
trained on ES-Topt, an evidence-supported dataset curated from primary
literature. VenusTOPT uses residue-level representations from the PRIME protein
language model and combines multi-scale local sequence-context decoding with
global protein-level feature decoding.

The model was further evaluated through experimental validation in a
temperature-guided endoglucanase mining workflow.

## Key Features

- Evidence-supported `Topt` training data
- Sequence-only inference
- Temperature-informed protein language model representations
- Multi-scale local and global sequence decoding
- Application to temperature-guided enzyme mining

## Repository Status

This repository is currently being prepared for public release. Source code,
pretrained model weights, inference scripts, and reproducibility resources will
be added during the manuscript review process.

## Planned Repository Structure

```text
VenusTOPT/
|-- inference/       # Sequence-to-Topt inference utilities
|-- models/          # VenusTOPT model definitions
|-- training/        # Model training and configuration scripts
|-- data/            # ES-Topt processing and split metadata
|-- benchmarks/      # Benchmark evaluation and metric scripts
|-- examples/        # Example sequences and expected predictions
|-- requirements.txt # Python dependencies
`-- README.md        # Project documentation
```

The final organization may be refined as the public release is prepared.

## Planned Release Contents

The public release is planned to include:

- VenusTOPT inference code
- Pretrained model checkpoints
- Training scripts
- ES-Topt dataset
- Benchmark evaluation scripts
- Example protein sequences and expected predictions
- Environment and dependency files

## Related Work

PRIME and VenusMine were developed in previous studies by our group.

- **PRIME** provides the temperature-informed protein language model representations used by VenusTOPT. PRIME: Jiang, F.;  Li, M.;  Dong, J.;  Yu, Y.;  Sun, X.;  Wu, B.;  Huang, J.;  Kang, L.;  Pei, Y.;  Zhang, L.;  Wang, S.;  Xu, W.;  Xin, J.;  Ouyang, W.;  Fan, G.;  Zheng, L.;  Tan, Y.;  Hu, Z.;  Xiong, Y.;  Feng, Y.;  Yang, G.;  Liu, Q.;  Song, J.;  Liu, J.;  Hong, L.; Tan, P., A general temperature-guided language model to design proteins of enhanced stability and activity. Science Advances 2024, 10 (48), eadr2641.
- **VenusMine** is the enzyme-mining framework into which VenusTOPT was integrated for experimental validation. VenusMine: Wu, B.;  Zhong, B.;  Zheng, L.;  Huang, R.;  Jiang, S.;  Li, M.;  Hong, L.; Tan, P., Harnessing protein language model for structure-based discovery of highly efficient and robust PET hydrolases. Nature Communications 2025, 16 (1), 6211.

## Citation

If you use VenusTOPT, please cite the associated manuscript. Citation
information will be updated upon publication.

```bibtex
@article{venusTOPT,
  title   = {VenusTOPT: Sequence-only prediction of enzyme optimal catalytic temperature},
  author  = {To be updated},
  journal = {To be updated},
  year    = {To be updated}
}
```

## License

License information will be added with the public code release.

## Contact

For questions regarding VenusTOPT, please contact:

- Author: Jin Huang
- Email: jinhuang.sjtu@gmail.com
