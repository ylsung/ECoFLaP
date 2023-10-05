# ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models

* Authors: [Yi-Lin Sung](https://ylsung.github.io/), [Jaehong Yoon](https://jaehong31.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* Paper: ["ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models"](https://arxiv.org/abs/2310.02998)
* [Project Page](https://ecoflap.github.io/)

we propose ECoFLaP, a two-stage coarse-to-fine weight pruning approach for Large Vision-Language Models (LVLMs). We first determine the sparsity ratios of different layers or blocks by leveraging the global importance score, which is efficiently computed based on the zeroth-order approximation of the global model gradients. Then, the multimodal model performs local layer-wise unstructured weight pruning based on the given ratios.

We validate our proposed method across various multimodal and unimodal models and datasets, demonstrating significant performance improvements over prevalent pruning techniques in the high-sparsity regime. 

![](assets/teaser.png)

## BLIP-2, FlanT5, ViT experiments

The main code for this part is in `LAVIS/`. Please do everything in LAVIS/ by `cd LAVIS/`.

### Installation

```
pip install -e .
```

### Dataset

Follow the scripts in `lavis/datasets/download_scripts/` to download the datasets.

### BLIP-2 Scripts

```bash

## BLIP-2 experiments

# ECoFLaP - zeroth order
python scripts/ecoflap_zeroth.py 0 12341

# ECoFLaP - first order
python scripts/ecoflap_first.py 0 12341

# Wanda
python scripts/wanda.py 0 12341

# SparseGPT
python scripts/sparsegpt.py 0 12341
```


## Bibtex

```bibtex
@inproceedings{Sung2023ECoFLaP,
    author = {Yi-Lin Sung, Jaehong Yoon, Mohit Bansal},
    title = {ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models},
    booktitle = {arXiv:2310.02998},
    year = {2023},
}
```