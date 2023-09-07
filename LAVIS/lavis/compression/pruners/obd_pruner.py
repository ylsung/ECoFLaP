import torch
from lavis.common.registry import registry

from lavis.compression.pruners.base_pruner import BasePruner
from lavis.compression.pruners.single_base_pruner import T5BasePruner, VITBasePruner
from lavis.compression.pruners.composable_base_pruner import BLIPT5BasePruner
from lavis.compression.pruners.utils import print_time


@registry.register_pruner("obd_pruner")
class OBDPruner(BasePruner):
    @print_time
    def compute_importance_scores(self, model, data_loader, loss_func, device, *args, **kwargs):
        org_device = list(model.parameters())[0].device
        
        # use GPU
        model.to(device)
        
        names, params = self.get_params(model)
        
        gradients_dict = {k: 0 for k in names}

        device = list(model.parameters())[0].device
        
        accum_samples = 0
        current_batch_index = 0
        
        for d in data_loader:
            print(accum_samples)
            if accum_samples >= self.num_samples:
                break
            
            loss, batch_len = loss_func(model, d, device != "cpu")

            accum_samples += batch_len
            current_batch_index += 1

            grads = torch.autograd.grad(loss, params)
            
            for k, v in zip(names, grads):
                gradients_dict[k] = v.cpu().data ** 2 # fisher

        for k in names:
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            gradients_dict[k] /= current_batch_index
                
        gradients_dict = self.fill_missing_scores(model, gradients_dict)
        
        # using square of magnitude multiplied by diagonal fisher as importance scores
        importance_measure = {k: (v.cpu().data ** 2) * gradients_dict[k] for k, v in self.model.state_dict().items()}
        
        model.to(org_device)

        return importance_measure


@registry.register_pruner("strct_obd_pruner")
class StrctOBDPruner(OBDPruner):
    def __init__(self, *args, **kwargs):
        kwargs["is_strct_pruning"] = True
        super().__init__(*args, **kwargs)


@registry.register_pruner("unstrct_obd_pruner")
class UnstrctOBDPruner(OBDPruner):
    def __init__(self, *args, **kwargs):
        kwargs["is_strct_pruning"] = False
        super().__init__(*args, **kwargs)
        

class OBDImportanceScore:
    @print_time
    def compute_importance_scores(self, model, data_loader, loss_func, device, *args, **kwargs):
        org_device = list(model.parameters())[0].device
        
        print(f"Use {device}")
        # use GPU
        model.to(device)
        
        names, params = self.get_params(model)
        
        gradients_dict = {k: 0 for k in names}

        device = list(model.parameters())[0].device
        
        accum_samples = 0
        current_batch_index = 0
        
        for d in data_loader:
            print(accum_samples)
            if accum_samples >= self.num_samples:
                break
            
            loss, batch_len = loss_func(model, d, device != "cpu")

            accum_samples += batch_len
            current_batch_index += 1

            grads = torch.autograd.grad(loss, params)
            
            for k, v in zip(names, grads):
                gradients_dict[k] = v.cpu().data ** 2 # fisher

        for k in names:
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            gradients_dict[k] /= current_batch_index
                
        gradients_dict = self.fill_missing_scores(model, gradients_dict)
        
        # using square of magnitude multiplied by diagonal fisher as importance scores
        importance_measure = {k: (v.cpu().data ** 2) * gradients_dict[k] for k, v in self.model.state_dict().items()}
        
        model.to(org_device)

        return importance_measure


@registry.register_pruner("t5_obd_pruner")
class T5OBDPruner(OBDImportanceScore, T5BasePruner):
    pruner_name = "t5_obd_pruner"


@registry.register_pruner("vit_obd_pruner")
class VITOBDPruner(OBDImportanceScore, VITBasePruner):
    pruner_name = "vit_obd_pruner"
    

@registry.register_pruner("blipt5_obd_pruner")
class BLIPT5OBDPruner(OBDImportanceScore, BLIPT5BasePruner):
    pruner_name = "blipt5_obd_pruner"