# ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models

The main code is in `/LAVIS`.

## Installation
```
cd LAVIS/
pip install -e .
```

## BLIP-2 Scripts

In `/LAVIS`

```bash
# ECoFLaP - zeroth order
python scripts/ecoflap_zeroth.py 0 12341

# ECoFLaP - first order
python scripts/ecoflap_first.py 0 12341

# Wanda
python scripts/wanda.py 0 12341

# SparseGPT
python scripts/sparsegpt.py 0 12341
```