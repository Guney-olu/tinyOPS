from pathlib import Path
from typing import List
import json
import numpy as np
from tinygrad import Context, Tensor, nn
from tinygrad.nn.state import load_state_dict
from tinyops.helper.helpers import concat_weights, load
from tinyops.helper.llama import Transformer, convert_from_huggingface, fix_bf16
from tinyops.helper.params import MODEL_PARAMS
from tinygrad.helpers import getenv

MAX_CONTEXT = getenv("MAX_CONTEXT", 4096)
# TODO add other models like mistral and gemma
class LLaMa:
    @staticmethod
    def build(model_path, tokenizer_path, model_gen="1", model_size="7B", quantize=None, device=None):
        params = MODEL_PARAMS[model_gen][model_size]
        tokenizer = MODEL_PARAMS[model_gen]['tokenizer'](model_file=str(tokenizer_path))
        assert tokenizer.vocab_size() == params["args"]["vocab_size"], f"{tokenizer.vocab_size()=} not equal to {params['args']['vocab_size']}"

        jit = bool(getenv("JIT", 1))

        if quantize == "int8":
            from tinyops.helper.quantization import Int8Linear as linear
        elif quantize == "nf4":
            from tinyops.helper.quantization import NF4Linear as linear
            linear = linear(64)
        else:
            linear = nn.Linear

        model = Transformer(**params["args"], linear=linear, max_context=MAX_CONTEXT, jit=jit)

        if model_path.is_dir():
            if (model_path / "model.safetensors.index.json").exists():
                weights = load(str(model_path / "model.safetensors.index.json"))
            elif (model_path / "model.safetensors").exists():
                weights = load(str(model_path / "model.safetensors"))
            else:
                weights = concat_weights([load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
        else:
            weights = load(str(model_path))
        
        if "model.embed_tokens.weight" in weights:
            weights = convert_from_huggingface(weights, model, 32, 4) #TODO no hardcoding
        
        weights = fix_bf16(weights)

        with Context(BEAM=0):
            if quantize is not None:
                weights = linear.quantize(weights, device)
                for _, v in weights.items(): 
                    v.realize()

            if isinstance(device, tuple):
                for k, v in nn.state.get_state_dict(model).items():
                    if 'scale' in k: v.shard_(device, axis=None)
                    elif '.attention.' in k: v.shard_(device, axis=-1)
                    elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
                    elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
                    elif '.feed_forward.' in k: v.shard_(device, axis=-1)
                    elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
                    elif 'output.weight' in k: v.shard_(device, axis=-1)
                    else: v.shard_(device, axis=None)

            load_state_dict(model, weights, strict=False, consume=True)

        return LLaMa(model, tokenizer)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    # def greedy_until(self, prompt: str, until, max_length, temperature):
    #     toks = [self.tokenizer.bos_id()] + self.tokenizer.encode(prompt)
    #     start_pos = 0
    #     for i in range(max_length):
    #         probs = self.model(Tensor([toks[start_pos:]]), start_pos, temperature).realize()
    #         probs_np = probs.numpy()
    #         tok = int(np.random.choice(len(probs_np), p=probs_np))
    #         start_pos = len(toks)
    #         toks.append(tok)

    #         if tok == self.tokenizer.eos_id(): 
    #             break
    #         output = self.tokenizer.decode(toks)
    #         for s in until:
    #             if output.endswith(s): 
    #                 return output[0:-len(s)]
    #     return output
    def greedy_until(self, prompt: str, until, max_length, temperature):
        toks = [self.tokenizer.bos_id()] + self.tokenizer.encode(prompt)
        start_pos = 0
        
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        for i in range(max_length):
            probs = self.model(Tensor([toks[start_pos:]]), start_pos, temperature).realize()
            probs_np = probs.numpy()
            probs_np = softmax(probs_np)  # Apply softmax to normalize probabilities
            tok = int(np.random.choice(len(probs_np), p=probs_np))
            start_pos = len(toks)
            toks.append(tok)

            if tok == self.tokenizer.eos_id(): 
                break
            output = self.tokenizer.decode(toks)
            for s in until:
                if output.endswith(s): 
                    return output[0:-len(s)]
        return output
