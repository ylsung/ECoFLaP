import torch
from time import time


def print_time(func):
    def wrapper(*args, **kwargs):
        start = time()

        ret = func(*args, **kwargs)
        
        time_spent = time() - start
        
        print(f"{func.__name__} spent {time_spent:.3f} s")
        
        return ret
    
    return wrapper


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


def loss_vision_language(model, samples, cuda_enabled):
    samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

    # samples = {key: s.half() for key, s in samples.items()}

    loss_dict = model(samples)
    loss = loss_dict["loss"]

    batch_len = len(samples["text_input"])

    return loss, batch_len


def loss_language(model, samples, cuda_enabled):
    samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

    # samples = {key: s.half() for key, s in samples.items()}

    loss_dict = model(samples)
    loss = loss_dict["loss"]

    batch_len = len(samples["text_input"])

    return loss, batch_len


def loss_vision(model, samples, cuda_enabled):
    # cross entropy loss
    samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
    outputs = model.predict(samples)

    logits = outputs["predictions"] / 100 # canceled out the multiplication by 100 in model.predict()
    targets = outputs["targets"]

    probs = torch.nn.functional.softmax(logits, -1)

    batch_index = torch.arange(len(targets)).to(targets.device)

    probs = probs[batch_index, targets]
    
    log_probs = probs.log()
    
    loss = - log_probs.mean()
    
    batch_len = len(targets)

    return loss, batch_len