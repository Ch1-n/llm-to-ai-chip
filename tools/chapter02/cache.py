"""Inference-only cache with explicit logical length and allocated capacity."""

import torch


class KVCache:
    def __init__(self, mode='cat', capacity=None):
        if mode not in ('cat', 'static'):
            raise ValueError(mode)
        if mode == 'static' and (capacity is None or capacity <= 0):
            raise ValueError('static cache needs a positive capacity')
        self.mode, self.capacity = mode, capacity
        self.length = 0
        self.storage = {}

    def append(self, **tensors):
        if torch.is_grad_enabled():
            raise ValueError('cache is inference-only; use no_grad or inference_mode')
        if not tensors or any(t.ndim != 4 for t in tensors.values()):
            raise ValueError('cache values must be rank-four tensors')
        size = next(iter(tensors.values())).shape[2]
        if size <= 0 or any(t.shape[2] != size for t in tensors.values()):
            raise ValueError('all cache fields need the same positive token count')
        if self.storage and set(tensors) != set(self.storage):
            raise ValueError('cache fields changed')
        end = self.length + size
        if self.capacity is not None and end > self.capacity:
            raise ValueError('cache capacity exceeded')
        for name, tensor in tensors.items():
            if name in self.storage:
                old = self.storage[name]
                if (old.shape[:2] + old.shape[3:] != tensor.shape[:2] + tensor.shape[3:]
                        or old.dtype != tensor.dtype or old.device != tensor.device):
                    raise ValueError('cache field shape, precision or device changed')
        if self.mode == 'cat':
            updated = {name: torch.cat((self.storage[name][:, :, :self.length], value), dim=2)
                       if name in self.storage else value
                       for name, value in tensors.items()}
        else:
            updated = dict(self.storage)
            for name, value in tensors.items():
                if name not in updated:
                    shape = list(value.shape)
                    shape[2] = self.capacity
                    updated[name] = value.new_empty(shape)
                updated[name][:, :, self.length:end].copy_(value)
        self.storage, self.length = updated, end
        return {name: value[:, :, :end] for name, value in updated.items()}

    def rewind(self, length):
        if not 0 <= length <= self.length:
            raise ValueError('cannot rewind beyond populated cache')
        self.length = length

    def logical_bytes(self):
        return sum(t.shape[0] * t.shape[1] * self.length * t.shape[3] * t.element_size()
                   for t in self.storage.values())

    def allocated_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage.values())
