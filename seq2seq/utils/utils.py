import os 
import regex as re
import logging
from dataclasses import fields
import torch.nn as nn
import json
from collections import OrderedDict


from transformers import (
    HfArgumentParser,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args(dataclasses, args_list=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(dataclasses)

    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args, petl_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args, petl_args = parser.parse_args_into_dataclasses()

    # args_list_from_default = [dataclass(output_dir=10) for dataclass in dataclasses]

    print(args_list)

    args_list_from_input = parser.parse_args_into_dataclasses(args_list)

    required_args_keys = ["--output_dir"]

    required_args = []
    for k in required_args_keys:
        required_args.append(k) # key
        required_args.append("None") # value

    args_list_from_default = parser.parse_args_into_dataclasses(required_args)

    training_args_from_input = args_list_from_input[2]


    if training_args_from_input.training_config_file is not None:
        args_list_from_config = parser.parse_json_file(json_file=training_args_from_input.training_config_file)
    else:
        args_list_from_config = args_list_from_default

    for i in range(len(dataclasses)):
        args_config = args_list_from_config[i]
        args_default = args_list_from_default[i]
        args_input = args_list_from_input[i]

        for k, v in vars(args_input).items():
            # the argument that is newly added from input
            # overwrite the value in config even it is assigned in config
            if v != getattr(args_default, k):
                setattr(args_config, k, v)

    return args_list_from_config


class IntermediateL2LossComputer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.side_features = OrderedDict()
        self.backbone_features = OrderedDict()

        for name, module in self.model.named_modules():

            if len(name.split(".")) > 1 and name.split(".")[-2] == "side_upsamples":
            # if any(n in name for n in ["side_upsamples", "final_side_layer_norm"]):
                module.register_forward_hook(self.get_side_output(name, self.side_features))
            elif "final_side_layer_norm" in name:
                module.register_forward_hook(self.get_side_output(name, self.side_features))

            if len(name.split(".")) > 1 and name.split(".")[-2] == "block":
            # if any(n in name for n in [".block", "final_layer_norm"]):
                module.register_forward_hook(self.get_output(name, self.backbone_features))
            elif "final_side_layer_norm" in name:
                module.register_forward_hook(self.get_output(name, self.backbone_features))
    
    def get_output(self, layer_name, features_dict):
        # for tuple output
        def hook(module, input, output):
            features_dict[layer_name] = output[0]
        return hook

    def get_side_output(self, layer_name, features_dict):
        def hook(module, input, output):
            features_dict[layer_name] = output
        return hook


def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_last_checkpoint(output_dir):
    if os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
        return output_dir
    return None


def pad_punctuation(text):
   """Re-implementation of _pad_punctuation in t5. This function adds spaces
   around punctuation. While this pads punctuation as expected, it has the 
   unexpected effected of padding certain unicode characters with accents, with
   spaces as well. For instance: "François" becomes "Fran ç ois"""
   # Pad everything except for: underscores (_), whitespace (\s),
   # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
   text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', text)
   # Collapse consecutive whitespace into one space.
   text = re.sub(r'\s+', ' ', text)
   return text


def num_parameters(model, petl_args):
    original_model_params_dict = {
        "t5-small": 60492288,
        "t5-base": 222882048,
        "t5-large": 737639424,
        "t5-3b": 2851569664,
    }    

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("***** Model Trainable Parameters {} *****".format(total_trainable_params))

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info("##### Parameter name %s", name)
    total_lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    
    total_trainable_bias_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.endswith(".bias"))
    total_trainable_layernorm_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ".layer_norm." in n)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total trainable parameters {total_trainable_params}")
    logger.info(f"Total traianable bias parameters {total_trainable_bias_params}")
    logger.info(f"Total trainable layernorm parameters {total_trainable_layernorm_params}")
    logger.info(f"Total parameters {total_params}")


    original_model_params = None # default values
    for key in original_model_params_dict.keys():
        if key in model.config._name_or_path:
            original_model_params = original_model_params_dict[key]

    if original_model_params is None:
        print("Didn't find a matched original model, so use the t5-base's size instead. Please make sure your model_name contain the keyword of t5-small, t5-base, t5-large, t5-3b.")
        original_model_params = 222882048

    # total params since we have 8 task, it is Y = 1*BERT + 8*ADAPTERS, and final number is Y/BERT ("1.3x")
    total_trainable_params_percent =(total_trainable_params/original_model_params)*100
    total_trainable_bias_params_percent =(total_trainable_bias_params/total_trainable_params)*100
    total_trainable_layernorm_params_percent =(total_trainable_layernorm_params/total_trainable_params)*100
    total_trainable_lm_head_params_percent =(total_lm_head_params/original_model_params)*100
    logger.info(f"Total trainable params percent {total_trainable_params_percent:.4f} (%)")
    logger.info(f"Total trainable bias params percent {total_trainable_bias_params_percent:.4f} (%)")
    logger.info(f"Total trainable layernorm params percent {total_trainable_layernorm_params_percent:.4f} (%)")
    logger.info(f"Total lm_head params percent {total_trainable_lm_head_params_percent:.4f} (%)")

    return total_trainable_params_percent


def save_json(filepath, dictionary):
    with open(filepath, "w") as outfile:
        json.dump(dictionary, outfile, indent=4)


def read_json(filepath):
    f = open(filepath,)
    return json.load(f)


def save_training_config(config_file_or_dict, output_dir):

    if isinstance(config_file_or_dict, dict):
        json_data = config_file_or_dict
    elif isinstance(config_file_or_dict, str):
        json_data = read_json(config_file)
    else:
        raise ValueError(f"config_file_or_dict should be either dict or str type, but found {type(config_file_or_dict)}")

    save_json(os.path.join(output_dir, "training_config.json"), json_data)

