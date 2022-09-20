from torch.nn.parallel import replicate, scatter, parallel_apply, gather
from torch.nn import DataParallel
import torch


def replicate_if_possible(network, devices=None):
    if devices is None:
        device_count = torch.cuda.device_count()
        devices = list(range(device_count))

    if len(devices) == 0:
        return [network]
    if len(devices) == 1:
        return [network.to(0)]
    return replicate(network, devices)


def scatter_if_needed(inputs, device_ids=None):
    if device_ids is None:
        device_count = torch.cuda.device_count()
        devices = list(range(device_count))
    if len(devices) == 0:
        return [inputs]
    if len(devices) == 1:
        return [inputs.to(0)]
    return scatter(inputs, devices)


def scatter_kwargs_if_needed(kwargs, device_ids=None):
    if device_ids is None:
        device_count = torch.cuda.device_count()
        devices = list(range(device_count))
    if len(devices) == 0:
        return [kwargs]
    if len(devices) == 1:
        return [{k: v.to(0) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}]
    return scatter(kwargs, devices)


def parallel_apply_if_needed(replicas, inputs, kwargs_tup=None):
    if len(replicas) == 1:
        return [replicas[0](*inputs[0], **({} if kwargs_tup is None else kwargs_tup[0]))]
    return parallel_apply(replicas, inputs, kwargs_tup)


def gather_if_needed(outputs, output_device=None):
    if len(outputs) == 1:
        return outputs[0]
    return gather(outputs, output_device or 0)
