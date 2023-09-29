"""
 Yi-Lin Sung https://ylsung.github.io/
"""

# from lavis.compression.woodfisher_pruner import WoodFisherPruner
from lavis.compression.pruners.base_pruner import BasePruner
from lavis.compression.pruners.single_base_pruner import SingleBasePruner
# from lavis.compression.pruners.mag_pruner import (
#     StrctMagPruner, 
#     UnstrctMagPruner, 
#     T5MagPruner, 
#     VITMagPruner,
#     BLIPT5MagPruner,
# )

# from lavis.compression.pruners.obd_pruner import (
#     StrctOBDPruner, 
#     UnstrctOBDPruner, 
#     T5OBDPruner, 
#     VITOBDPruner,
#     BLIPT5OBDPruner,
# )

from lavis.compression.pruners.global_pruner import (
    BLIPT5MagPruner,
    BLIPT5AOBDPruner,
    BLIPT5AMeZoPruner,
)

from lavis.compression.pruners.wanda_pruner import (
    T5LayerWandaPruner, VITLayerWandaPruner, BLIPT5LayerWandaPruner
)
from lavis.compression.pruners.sparsegpt_pruner import (
    T5LayerSparseGPTPruner, VITLayerSparseGPTPruner, BLIPT5LayerSparseGPTPruner
)
from lavis.common.registry import registry

from omegaconf import OmegaConf

__all__ = [
    # "WoodFisherPruner",
    "BasePruner",
]


def load_pruner(name, model, data_loader, cfg_path=None, cfg=None):
    
    if cfg_path is None and cfg is None:
        cfg = None
    elif cfg_path is not None:
        cfg = OmegaConf.load(cfg_path)

    try:
        pruner = registry.get_pruner_class(name)(model=model, data_loader=data_loader, **cfg)

    except TypeError:
        print(
            f"Pruner {name} not found. Available pruners:\n"
            + ", ".join([str(k) for k in __all__])
        )
        exit(1)

    return pruner