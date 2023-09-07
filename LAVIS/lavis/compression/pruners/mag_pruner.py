from lavis.common.registry import registry

from lavis.compression.pruners.base_pruner import BasePruner
from lavis.compression.pruners.single_base_pruner import T5BasePruner, VITBasePruner
from lavis.compression.pruners.composable_base_pruner import BLIPT5BasePruner
from lavis.compression.pruners.utils import print_time


@registry.register_pruner("mag_pruner")
class MagPruner(BasePruner):
    @print_time
    def compute_importance_scores(self, *args, **kwargs):
        # using square of magnitude as the measure.
        importance_measure = {k: v ** 2 for k, v in self.model.state_dict().items()}
        return importance_measure
    
    
@registry.register_pruner("strct_mag_pruner")
class StrctMagPruner(MagPruner):
    def __init__(self, *args, **kwargs):
        kwargs["is_strct_pruning"] = True
        super().__init__(*args, **kwargs)


@registry.register_pruner("unstrct_mag_pruner")
class UnstrctMagPruner(MagPruner):
    def __init__(self, *args, **kwargs):
        kwargs["is_strct_pruning"] = False
        super().__init__(*args, **kwargs)
        
        
class MagImportanceScore:
    @print_time
    def compute_importance_scores(self, *args, **kwargs):
        # using square of magnitude as the measure.
        importance_measure = {k: v ** 2 for k, v in self.model.state_dict().items()}
        return importance_measure


@registry.register_pruner("t5_mag_pruner")
class T5MagPruner(MagImportanceScore, T5BasePruner):
    pruner_name = "t5_mag_pruner"


@registry.register_pruner("vit_mag_pruner")
class VITMagPruner(MagImportanceScore, VITBasePruner):
    pruner_name = "vit_mag_pruner"


@registry.register_pruner("blipt5_mag_pruner")
class BLIPT5MagPruner(MagImportanceScore, BLIPT5BasePruner):
    pruner_name = "blipt5_mag_pruner"

