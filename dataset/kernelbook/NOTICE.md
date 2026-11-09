# KernelBook-derived problems

The PyTorch modules in `selection.jsonl` come from KernelBook (GPU MODE, https://huggingface.co/datasets/GPUMODE/KernelBook,
`dataset_permissive.parquet`), used under the June 9 Researcher Reciprocity License (`LICENSE.txt`, unmodified). Each module also
keeps its original repository's license, checked at the pinned commit (`license_checked`, `repo_link`).

Modified: the JAX references under `problems/kernelbook/` are ports of these modules written by the dataset's constructor, and
`port_kernelbook.py` appends a KernelBench-format adapter to each module before porting. Attachment A of the license applies to any
model trained on rows derived from this data: researchers must be able to generate with, evaluate and publish about it on the same
terms as other users.
