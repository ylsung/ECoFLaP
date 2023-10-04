import torch
from time import time
from lavis.datasets.data_utils import prepare_sample


def print_time(func):
    def wrapper(*args, **kwargs):
        start = time()

        ret = func(*args, **kwargs)
        
        time_spent = time() - start
        
        print(f"{func.__name__} spent {time_spent:.3f} s")
        
        return ret
    
    return wrapper


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