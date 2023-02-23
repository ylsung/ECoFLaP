import os
import argparse
from seq2seq.utils import parse_args
from seq2seq.training_args import TrainingArguments, ModelArguments, DataTrainingArguments, PETLModelArguments

DIR = os.path.dirname(__file__)

def test_argparser():
    json_file = os.path.join(DIR, "test.json")
    

    args_list = [
        "--training_config_file", f"{json_file}", 
        "--save_steps", "700", 
        "--adapter_reduction_factor", "87",
        "--output_dir", "input",
    ]

    model_args, data_args, training_args, petl_args \
        = parse_args([ModelArguments, DataTrainingArguments, TrainingArguments, PETLModelArguments], args_list)

    assert model_args.model_name_or_path == "test" # config value

    assert training_args.max_steps == -1 # default value
    assert training_args.save_steps == 700 # input value
    assert training_args.seed == 99 # config value

    assert petl_args.trainable_param_names == "*" # default value
    assert petl_args.adapter_reduction_factor == 87 # input value
    assert petl_args.adapter_non_linearity == "relu" # config value

    assert training_args.output_dir == "input"


    args_list = [
        "--save_steps", "700", 
        "--adapter_reduction_factor", "87",
        "--output_dir", "input",
    ]

    model_args, data_args, training_args, petl_args \
        = parse_args([ModelArguments, DataTrainingArguments, TrainingArguments, PETLModelArguments], args_list)

    assert training_args.max_steps == -1 # default value
    assert training_args.save_steps == 700 # input value
    assert training_args.seed == 42 # default value

    assert petl_args.trainable_param_names == "*" # default value
    assert petl_args.adapter_reduction_factor == 87 # input value
    assert petl_args.adapter_non_linearity == "swish" # default value

    assert training_args.output_dir == "input"

    
if __name__ == "__main__":
    test_argparser()
