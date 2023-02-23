from dataclasses import dataclass, field
from typing import Optional, List
from transformers import Seq2SeqTrainingArguments


# run_seq2seq parameters.
@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    training_config_file: Optional[str] = field(
        default=None, metadata={"help": "Hyper-parameters for the training"}
    )
    print_num_parameters: Optional[bool] = field(default=False, metadata={"help": "If set, print the parameters of "
                                                                                 "the model."})
    do_test: Optional[bool] = field(default=False, metadata={"help": "If set, evaluates the test performance."})
    split_validation_test: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    compute_time: Optional[bool] = field(default=False, metadata={"help": "If set measures the time."})
    compute_memory: Optional[bool] = field(default=False, metadata={"help": "if set, measures the memory"})
    prefix_length: Optional[int] = field(default=100, metadata={"help": "Defines the length for prefix tuning."})
    add_prefix_to_inputs: Optional[bool] = field(default=True, metadata={"help": "if set, add task name in front of the input."})
    trainable_encoder_layers: Optional[List[int]] = field(default=None, metadata={"help": "Defines the encoder layers id"
                                                                                      "in which parameters are trainable"})
    trainable_decoder_layers: Optional[List[int]] = field(default=None, metadata={"help": "Defines the decoder layers id"
                                                                                      "in which parameters are trainable"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    weight_to_load: str = field(
        default=None,
        metadata={
            "help": "The path for the trained weight."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    eval_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the evaluation dataset to use (via the datasets library)."}
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the evaluation dataset to use (via the datasets library)."}
    )
    test_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the test dataset to use (via the datasets library)."}
    )
    test_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the test dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    task_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from task adapters to the tasks."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."}
    )

    use_train_as_validation: Optional[bool] = field(
        default=False,
        metadata={"help": "Use train split for validation. Only for sanity check if the model can at least overfit on train split."}
    )

    # def __post_init__(self):
    #     if self.task_name is None:
    #         raise ValueError("Need either a dataset name or a training/validation file.")
    #     if self.val_max_target_length is None:
    #         self.val_max_target_length = self.max_target_length
    #     if self.test_max_target_length is None:
    #         self.test_max_target_length = self.max_target_length


@dataclass
class PETLModelArguments:
    """Defines the hyper-parameters for petl methods."""
    # general arguments
    petl_method: str = field(default=None, metadata={
        "help": "which petl methods are going to use, should be one of [adapter, lora, bitfit, prompt-tuning, prefix-tuning, lst]"})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "If set unfreeze the last linear layer."})
    freeze_lm_head: bool = field(default=False, metadata={"help": "If set, freeze the last linear layer."})
    unfreeze_layer_norms: bool = field(default=False, metadata={"help": "If set, unfreezes the layer norms."})
    trainable_param_names: str = field(default=".*", metadata={"help": "The names of the parameters that are going to be trained. It follows the regular expression format."})

    # adapter configuration
    hidden_dim: Optional[int] = field(default=128, metadata={"help": "defines the default hidden dimension for "
        "adapter layers."})
    adapter_reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
        "adapter layers."})
    adapter_non_linearity: Optional[str] = field(default="swish", metadata={"help": "Defines nonlinearity for adapter layers."})
    
    # adapter_layers_encoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
    #                                                                                   "in which task adapters is"
    #                                                                                   "added in the encoder."})
    # adapter_layers_decoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
    #                                                                                   "in which task adapters is"
    #                                                                                   "added in the decoder."})
    # adapter_in_decoder: Optional[bool] = field(default=True, metadata={"help": "If set to false, do not include"
    #                                                                                 "task adapters in the decoder."})
    # add_adapter_in_feed_forward: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the feedforward."})
    # add_adapter_in_self_attention: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the selfattention"})

    # lora
    lora_rank: Optional[int] = field(default=16, metadata={"help": "The rank of the lora linear layers."})
    lora_init_scale: Optional[float] = field(default=0.01, metadata={"help": "The value to scale the initialization weights."})
    lora_scaling_rank: Optional[int] = field(default=1, metadata={"help": "The rank for multilora."})
    lora_modules: Optional[str] = field(default=".*SelfAttention|.*EncDecAttention", metadata={"help": "The modules where loras are added"})
    lora_layers: Optional[str] = field(default="q|k|v|o", metadata={"help": "The layers where loras are added"})

    # prefix-tuning
    prefix_tuning: Optional[bool] = field(default=False, metadata={"help": "If set, uses prefix tuning."})
    prefix_dim: Optional[int] = field(default=100, metadata={"help": "Specifies the prefix embedding dimension."})
    init_prefix_from_vocab: Optional[bool] = field(default=False, metadata={"help": "Initialize prefix from the tokens of pretrained t5-base model."})

    # bitfit
    bitfit: Optional[bool] = field(default=False, metadata={"help": "If set, we train the bitfit model."})
    freeze_bitfit_lm_head: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})
    freeze_bitfit_lm_head_all: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})

    # ladder side-tuning
    lst_reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
        "lst layers."})
    lambda_distill: float = field(default=0, metadata={"help": "The weight for distill loss for pre-training the side network."})
    lambda_label: float = field(default=1, metadata={"help": "The weight for label loss for pre-training the side network."})
    lambda_kd_ir: float = field(default=1, metadata={"help": "The weight for balancing kd and ir loss (only used in LIT distillation)."})
    lit_distillation: bool = field(default=False, metadata={"help": "Whether to use LIT distillation."})
    train_t5_mlm: bool = field(default=False, metadata={"help": "If set, use t5 mlm data collator and use t5 mlm objective to pre-train the side network"})
    mlm_ratio: float = field(default=0.15, metadata={"help": "The masking ratio for MLM objective"}) 

    gate_T: float = field(default=0.1, metadata={"help": "The temperature for gates."})
    gate_alpha: float = field(default=0.0, metadata={"help": "The initial parameter for gating"})
    gate_type: str = field(default="learnable", metadata={"help": "What type of gate is used. Should be one from [learnable, all_from_side, all_from_backbone, avg]"})

    load_side_pretrained_weights: str = field(default="", metadata={"help": "The intial weights for side network."})
    encoder_side_layers: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which side transformer is"
                                                                                      "added in the encoder."})
    decoder_side_layers: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which side transformer is"
                                                                                      "added in the decoder."})

    create_side_lm: bool = field(default=False, metadata={"help": "If set, create lm_head and embedding layers for the side network."})
    freeze_side_lm: bool = field(default=False, metadata={"help": "If set, freeze the lm_head and embedding layers of the side network."})

    samples_for_fisher: Optional[int] = field(default=1024, metadata={"help": "How many samples are used to compute fisher information?"})

    merge_last: bool = field(default=False, metadata={"help": "If set, merge the information after the last layer of the backbone and side network."})


    # prompt by pre-trained model
    pbp_reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
        "pbp layers."})
    enc_num_tokens: Optional[int] = field(default=10, metadata={"help": "Specifies the prefix embedding dimension in encoder."})
    dec_num_tokens: Optional[int] = field(default=10, metadata={"help": "Specifies the prefix embedding dimension in decoder."})
    prompts_expand_after: Optional[bool] = field(default=False, metadata={"help": "Specifies the prefix embedding dimension in decoder."})
    init_from_emb: Optional[bool] = field(default=False, metadata={"help": "Whether to initialize the prompt from the tokenizer's embeddings."})
    hard_prompt: Optional[str] = field(default=None, metadata={"help": "The hard prompt for the backbone model."})