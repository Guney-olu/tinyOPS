from pathlib import Path
from typing import List
import json
from tinygrad import Tensor
from tinygrad.nn.state import safe_load, torch_load

def concat_weights(models, device=None):
    def convert(name) -> Tensor:
        disk_tensors: List[Tensor] = [model[name] for model in models]
        if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
            return disk_tensors[0].to(device=device)
        axis = 1 if name.startswith("tok_embeddings.") or name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
        lazy_tensors = [data.to(device=device) for data in disk_tensors]
        return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
    return {name: convert(name) for name in {name: None for model in models for name in model}}

def load(fn: str):
    if fn.endswith('.index.json'):
        with open(fn) as fp: 
            weight_map = json.load(fp)['weight_map']
        parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
        return {k: parts[n][k] for k, n in weight_map.items()}
    elif fn.endswith(".safetensors"):
        return safe_load(fn)
    else:
        return torch_load(fn)
