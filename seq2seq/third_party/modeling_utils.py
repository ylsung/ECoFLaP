import collections
import gc
import inspect
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from transformers.utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from transformers.utils.import_utils import ENV_VARS_TRUE_VALUES, is_sagemaker_mp_enabled

from transformers.activations import get_activation
from transformers.configuration_utils import PretrainedConfig
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from transformers.dynamic_module_utils import custom_object_save
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.pytorch_utils import (  # noqa: F401
    Conv1D,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
    prune_layer,
    prune_linear_layer,
)
from transformers.utils import (
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    ModelOutput,
    PushToHubMixin,
    cached_file,
    copy_func,
    download_url,
    has_file,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_offline_mode,
    is_remote_url,
    is_safetensors_available,
    is_torch_tpu_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.versions import require_version_core

from third_party.trainers.seq2seq_trainer import remove_teacher_from_state_dict


XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    from accelerate.utils import (
        load_offloaded_weights,
        offload_weight,
        save_offload_index,
        set_module_tensor_to_device,
    )

    if version.parse(accelerate_version) > version.parse("0.11.0"):
        from accelerate.utils import get_balanced_memory
    else:
        get_balanced_memory = None

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file

logger = logging.get_logger(__name__)


_init_weights = True


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


from transformers.modeling_utils import unwrap_model, get_parameter_dtype


def save_pretrained(
    self,
    save_directory: Union[str, os.PathLike],
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    push_to_hub: bool = False,
    max_shard_size: Union[int, str] = "10GB",
    safe_serialization: bool = False,
    **kwargs,
):
    """
    Save a model and its configuration file to a directory, so that it can be re-loaded using the
    [`~PreTrainedModel.from_pretrained`] class method.
    Arguments:
        save_directory (`str` or `os.PathLike`):
            Directory to which to save. Will be created if it doesn't exist.
        is_main_process (`bool`, *optional*, defaults to `True`):
            Whether the process calling this is the main process or not. Useful when in distributed training like
            TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
            the main process to avoid race conditions.
        state_dict (nested dictionary of `torch.Tensor`):
            The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
            save parts of the model or if special precautions need to be taken when recovering the state dictionary
            of a model (like when using model parallelism).
        save_function (`Callable`):
            The function to use to save the state dictionary. Useful on distributed training like TPUs when one
            need to replace `torch.save` by another method.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
            repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
            namespace).
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
            lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
            <Tip warning={true}>
            If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
            which will be bigger than `max_shard_size`.
            </Tip>
        safe_serialization (`bool`, *optional*, defaults to `False`):
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
        kwargs:
            Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
    """
    # Checks if the model has been loaded in 8-bit
    if getattr(self, "is_loaded_in_8bit", False):
        warnings.warn(
            "You are calling `save_pretrained` to a 8-bit converted model you may likely encounter unexepected"
            " behaviors. ",
            UserWarning,
        )

    if "save_config" in kwargs:
        warnings.warn(
            "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
        )
        is_main_process = kwargs.pop("save_config")
    if safe_serialization and not is_safetensors_available():
        raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

    if os.path.isfile(save_directory):
        logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
        return

    os.makedirs(save_directory, exist_ok=True)

    if push_to_hub:
        commit_message = kwargs.pop("commit_message", None)
        repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
        repo_id, token = self._create_repo(repo_id, **kwargs)
        files_timestamps = self._get_files_timestamps(save_directory)

    # Only save the model itself if we are using distributed training
    model_to_save = unwrap_model(self)

    # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
    # we currently don't use this setting automatically, but may start to use with v5
    dtype = get_parameter_dtype(model_to_save)
    model_to_save.config.torch_dtype = str(dtype).split(".")[1]

    # Attach architecture to the config
    model_to_save.config.architectures = [model_to_save.__class__.__name__]

    # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
    # loaded from the Hub.
    if self._auto_class is not None:
        custom_object_save(self, save_directory, config=self.config)

    # Save the config
    if is_main_process:
        model_to_save.config.save_pretrained(save_directory)
        # if self.can_generate():
        # HACKT WAY
        model_to_save.generation_config.save_pretrained(save_directory)

    # Save the model
    if state_dict is None:
        state_dict = model_to_save.state_dict()

    state_dict = remove_teacher_from_state_dict(state_dict)

    # Translate state_dict from smp to hf if saving with smp >= 1.10
    if IS_SAGEMAKER_MP_POST_1_10:
        for smp_to_hf, _ in smp.state.module_manager.translate_functions:
            state_dict = smp_to_hf(state_dict)

    # Handle the case where some state_dict keys shouldn't be saved
    if self._keys_to_ignore_on_save is not None:
        for ignore_key in self._keys_to_ignore_on_save:
            if ignore_key in state_dict.keys():
                del state_dict[ignore_key]

    # Shard the model if it is too big.
    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

    # Clean the folder from a previous save
    for filename in os.listdir(save_directory):
        full_filename = os.path.join(save_directory, filename)
        # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
        # in distributed settings to avoid race conditions.
        weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")
        if (
            filename.startswith(weights_no_suffix)
            and os.path.isfile(full_filename)
            and filename not in shards.keys()
            and is_main_process
        ):
            os.remove(full_filename)

    # Save the model
    for shard_file, shard in shards.items():
        if safe_serialization:
            # At some point we will need to deal better with save_function (used for TPU and other distributed
            # joyfulness), but for now this enough.
            safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
        else:
            save_function(shard, os.path.join(save_directory, shard_file))

    if index is None:
        logger.info(f"Model weights saved in {os.path.join(save_directory, WEIGHTS_NAME)}")
    else:
        save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        save_index_file = os.path.join(save_directory, save_index_file)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger.info(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

    if push_to_hub:
        self._upload_modified_files(
            save_directory, repo_id, files_timestamps, commit_message=commit_message, token=token
        )