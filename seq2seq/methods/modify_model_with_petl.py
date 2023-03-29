from seq2seq.methods.adapters.modify_model_with_adapter import t5_modify_with_adapters
from seq2seq.methods.lora.modify_model_with_lora import modify_with_lora
from seq2seq.methods.lst.modify_model_with_lst import t5_modify_with_lst
from seq2seq.methods.pbp_extra_input.modify_model_with_pbp import t5_modify_with_pbp
from seq2seq.methods.vanilla.modify_model_with_vanilla import t5_modify_with_vanilla 
from seq2seq.methods.prefix_tuning.modify_model_with_prefix_tuning import t5_modify_with_prefix_tuning 
from seq2seq.methods.weight_init.modify_model_with_weight_init import t5_modify_with_weight_init


modify_functions = {
    "adapter": t5_modify_with_adapters,
    "lora": modify_with_lora,
    "lst": t5_modify_with_lst,
    "pbp": t5_modify_with_pbp,
    "vanilla": t5_modify_with_vanilla,
    "prefix_tuning": t5_modify_with_prefix_tuning,
    "weight_init": t5_modify_with_weight_init,
}

def modify_model_with_petl(transformer, petl_config, *args, **kwargs):
    if petl_config.petl_method == None:
        return transformer

    for method in petl_config.petl_method.split("|"):
        assert method in modify_functions, f"{method} is not implemented, please use one of {list(modify_functions.keys())}"
        transformer = modify_functions[method](transformer, petl_config, *args, **kwargs)

    return transformer
