# ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models (ICLR 2024)

* Authors: [Yi-Lin Sung](https://ylsung.github.io/), [Jaehong Yoon](https://jaehong31.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* Paper: ["ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models"](https://arxiv.org/abs/2310.02998)
* [Project Page](https://ecoflap.github.io/)

We propose ECoFLaP, a two-stage coarse-to-fine weight pruning approach for Large Vision-Language Models (LVLMs). We first determine the sparsity ratios of different layers or blocks by leveraging the global importance score, which is efficiently computed based on the zeroth-order approximation of the global model gradients. Then, the multimodal model performs local layer-wise unstructured weight pruning based on the given ratios.

We validate our proposed method across various multimodal and unimodal models and datasets, demonstrating significant performance improvements over prevalent pruning techniques in the high-sparsity regime. 

![](assets/teaser.png)

## Changelog

* [Feb 2024] Checkpoints are added

## Checkpoints

### BLIP2

Sparsities are all 0.5

| Wanda | ECoFLaP first-order   | ECoFLaP zeroth-order |
|------------|------------| ------------|
| [Ckpt](https://drive.google.com/file/d/1XQE0YalFgwCM6FjCWZu-PmFpEkpUU4i5/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1GAJ-4Aln4IHCQmpCNIRJ4AkhwCAEgcAg/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1HOqyltl7KmJ1lDk8JUbt2no0mY_4uaWF/view?usp=drive_link) |

### FlanT5 XL

| Wanda | ECoFLaP first-order   | ECoFLaP zeroth-order |
|------------|------------| ------------|
| [Ckpt](https://drive.google.com/file/d/1P3hNvH71f7f1sm8hXBIyevGCvCR8jqAV/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1MncmHfJXPIrzO65VQIk4c0EJvRgEEmFt/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1P52fhVDIm95J_DX8iNV6eRjLsxSHEbqB/view?usp=drive_link) |

### ViT b/16

| Wanda | ECoFLaP first-order   | ECoFLaP zeroth-order |
|------------|------------| ------------|
| [Ckpt](https://drive.google.com/file/d/1nzDktg7Y6cTIhCSXyRWp2dMggXGihRQP/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1o_UiCcLfzUuUGlAjcxQEv2yzPNYQuOK3/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/17wJVCFUUTb77UsnYoQ6YyMHXugw9OpFN/view?usp=drive_link) |


## CLIP

Sparsities are all 0.4

| Wanda | SparseGPT | ECoFLaP w/ Wanda | ECoFLaP w/ SparseGPT |
|------------|------------|------------|------------|
| [Ckpt](https://drive.google.com/file/d/1wvCWG0NnE3-Tlvdry7_XoeYlzaiu-ZUQ/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1bcZb-vu_25hloAkW1r1vgcND41jlf15w/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1cOq_wXOiIfIPu7dCrlqMRuSmFaoClYR-/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1pEPklf3Bm4uVmZHdogdQ1Kr_J2oP3rhi/view?usp=drive_link) |


## BLIP

Sparsities are all 0.5

| Dataset | Wanda | ECoFLaP | ECoFLaP w/ fine-tuning |
|------------|------------|------------|------------|
| VQA | [Ckpt](https://drive.google.com/file/d/1IPcKvhkHGDhlZLSRlGlZR4h6HMHU3iNM/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1l8noCREc3-DrtBaZ3AqVkxcPqh8tWkmB/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/117xDKd4iZ_lZXt6NdK0dPUfcaMk_Yh9L/view?usp=drive_link) |
| NLVR2 | [Ckpt](https://drive.google.com/file/d/1-3CzPIH2v-axciO3lnl2RRcT24GI0GIp/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1-pWH0OVT_bdKLpiO3scMz7zBKqj7_lx1/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/123qQpm7XL3D7jSbjpIlXdMIaadTU9-Er/view?usp=drive_link) |
| Flickr | [Ckpt](https://drive.google.com/file/d/1AiX2m1Itdi5_eQf8Rcl7QfNcquuHfE8j/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/14dOt7VLZMM0YfKPJNT9vHUBXNGOAqHBM/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/17E0qem-APvzN5UjN9FruqcaPtwhX2Qd4/view?usp=drive_link) |
| COCO Caption | [Ckpt](https://drive.google.com/file/d/1ji0e1GUlfY1mh8vXrK2N4m3IkaOXh5GC/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1mOHboHiRPpRoUCKu96jsUsMzgepVdUTX/view?usp=drive_link) | [Ckpt](https://drive.google.com/file/d/1VMpITHZpvqgq27Hc6BJ3tAMUxdXGgX64/view?usp=drive_link) |

* Some additional results regarding BLIP models


|Methods | VQA (test dev) | Flickr30k (TR@1/IR@1) | NLVR2 (val/text) | COCO Cap. (CIDEr/SPICE) |
|------------|------------|------------|------------|------------|
|Full model | 77.4 | 96.8/86.9 | 82.3/83.6 | 133.3/23.8 |
|Wanda (w/o fine-tuning) | 71.9 | 85.3/72.3 |  78.3/78.1 | 97.1/18.4 |
|ECoFLaP (w/o fine-tuning) | 73.6 |	90.2/79.5 | 79.1/79.2 | 111.0/20.3 |
|UPop (w/ fine-tuning) | 76.3| 94.0/82.0 | 80.3/81.1 | 128.9/23.3| 
|ECoFLaP (w/ fine-tuning) | 76.7 |	96.8/85.6 | 81.8/82.5 | 132.3/23.8 |



## BLIP-2, FlanT5, ViT experiment scripts

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
python scripts/blip2/ecoflap_zeroth.py 0 12341

# ECoFLaP - first order
python scripts/blip2/ecoflap_first.py 0 12341

# Wanda
python scripts/blip2/wanda.py 0 12341

# SparseGPT
python scripts/blip2/sparsegpt.py 0 12341
```

### ViT Scripts

```bash
# ECoFLaP - zeroth order
python scripts/eva_clip/ecoflap.py 0 12341

# Wanda
python scripts/eva_clip/wanda.py 0 12341
```

### FlanT5 Scripts

```bash
### Generate the pruned checkpoint

# ECoFLaP - zeroth order
python scripts/t5/ecoflap.py 0 12341

### Do the five-shot evaluation

# go to the mmlu_eval folder
cd ../mmlu_eval

# Make sure to assign pruned_checkpoint to the checkpoint generated in the previous step
bash test.sh 
```

## CLIP experiments

The main code for this part is in `CoOp/`. Please do everything in CoOp/ by `cd CoOp/`.

### Installation

```
pip install -r requirements.txt
```

### Dataset

Follow the scripts in `DATASETS.md` to download the datasets.

### Scripts

```bash
# Wanda and ECoFLaP (w/ Wanda)
bash scripts/coop/ecoflap_wanda.sh

# SparseGPT and ECoFLaP (w/ SparseGPT)
bash scripts/coop/ecoflap_sparsegpt.sh
```


## BLIP experiments to compare with UPop

The main code for this part is in `UPop/`. Please do everything in UPop/ by `cd UPop/`.

### Installation

```
pip install -r requirements.txt
```

### Dataset

Follow the scripts in `README.md` to download the datasets.

### Scripts

```bash
### task=coco, flickr, nlvr2, vqa

# Wanda
bash ecoflap_scripts/${task}/wanda.sh

# ECoFLaP
bash ecoflap_scripts/${task}/ecoflap.sh

# Fine-tune the pruned checkpoint obtained by ECoFLaP
bash ecoflap_scripts/${task}/ecoflap_finetuning.sh
```


## LLaMA experiments

The main code for this part is in `LLaMA/`. Please do everything in LLaMA/ by `cd LLaMA/`.

### Installation

Follow the scripts in `Install.md`.


### Scripts

I removed `--cache_dir` so the program will read the cache that store in `$HF_HOME$` (if specified) or default cache directory.

```bash
# ECoFLaP
bash scripts/ecoflap_zero.sh 0
```


## Bibtex

```bibtex
@inproceedings{Sung2024ECoFLaP,
    author = {Yi-Lin Sung, Jaehong Yoon, Mohit Bansal},
    title = {ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2024},
}
```