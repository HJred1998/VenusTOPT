# Data availability

ES-Topt is not included at this stage. No dataset sequences, experimental
labels, accession lists, split assignments, or wet-lab measurements are bundled.
Release information will be added during the publication process.

User-supplied training CSV schema:

```text
uniprot_id,topt,sequence
```

`sequence` is optional. Targets must be numeric degrees Celsius; unique IDs
must match `<uniprot_id>.npy` embeddings of shape `[L,1280]`. Keep train,
validation, and test proteins disjoint. Store private inputs in `data/private/`,
which is excluded by `.gitignore`.

Example FASTA sequences illustrate the input format. They are artificial
sequences and have no experimental Topt labels.
