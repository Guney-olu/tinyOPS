from pathlib import Path
import functools
from typing import List
import numpy as np
from tinygrad import Context, Tensor, nn,GlobalCounters,Device,Variable
from tinygrad.nn.state import get_state_dict, load_state_dict,get_parameters,torch_load
from tinyops.helper.helpers import concat_weights, load
from tinyops.helper.llama import FeedForward,Transformer, convert_from_huggingface, fix_bf16
from tinyops.helper.params import MODEL_PARAMS
from tinygrad.helpers import getenv,Timing, Profiling,DEBUG,CI,tqdm

MAX_CONTEXT = getenv("MAX_CONTEXT", 4096)
# TODO add other models like mistral and gemma
class LLaMa:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
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
        elif quantize=="nf2":
            from tinyops.helper.quantization import NF2Linear as linear
            linear = linear(16)
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
            weights = convert_from_huggingface(weights, model, params["args"]["n_heads"], params["args"]["n_kv_heads"]) #TODO no hardcoding
        
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

    @staticmethod
    def param_bytes(llama):
        return sum(x.lazydata.size * x.dtype.itemsize for x in get_parameters(llama.model))

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
    
    @staticmethod
    def generate(llama,max_tokens,prompt,temperature,device):
        import sys
        outputted = prompt
        start_pos, toks = 0, [llama.tokenizer.bos_id()] + llama.tokenizer.encode(outputted)

        for _ in range(max_tokens):
            tok_tensor = llama.model(Tensor([toks[start_pos:]], device=device), start_pos, temperature)
            tok = tok_tensor.item()
            start_pos = len(toks)
            toks.append(tok)
            print(toks)
            cur = llama.tokenizer.decode(toks)
            sys.stdout.write(cur[len(outputted):])
            sys.stdout.flush()
            outputted = cur

        return outputted


class mixtral:
    def __init__(self, num_experts:int, dim:int, hidden_dim:int, linear=nn.Linear):
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = [FeedForward(dim, hidden_dim, linear) for _ in range(num_experts)]
    def __call__(self, x:Tensor) -> Tensor:
        assert x.shape[0] == 1, "only BS=1"
        g = self.gate(x).float().exp()
        choice = g.data().tolist()[0][0]
        top = sorted(enumerate(choice), key=lambda x: -x[1])
        norm = top[0][1] + top[1][1]
        e1, e2 = self.experts[top[0][0]], self.experts[top[1][0]]
        scale = Tensor([top[0][1]/norm, top[1][1]/norm])
        ret = e1(x.to(e1.w1.weight.device)).to(x.device) * scale[0] + \
            e2(x.to(e2.w1.weight.device)).to(x.device) * scale[1]
        return ret
    
    @staticmethod
    def build(weight_path):
        state = torch_load(weight_path + "/consolidated.00.pth.b")
        model = Transformer(n_layers=32, dim=4096, hidden_dim=14336, n_heads=32, n_kv_heads=8, norm_eps=1e-5, vocab_size=32000, feed_forward=functools.partial(mixtral, 8), jit=False)
        model_state_dict = get_state_dict(model)
        
        for k in (t := tqdm(state, disable=CI)):
            if 'feed_forward.experts.' in k:
                expert_no = int(k.split('feed_forward.experts.')[1].split('.')[0])
                device = Device.DEFAULT + ":" + str((expert_no//2)+1)
            else:
                device = Device.DEFAULT
        t.set_description(f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, loading {k} to {device}")
        model_state_dict[k].replace(state[k].to(device).half()).realize()
        if CI: print(f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB")
        return model
    
    @staticmethod
    def generate(model,tokenzier_path,max_tokens,prompt,temperature,device):
        from sentencepiece import SentencePieceProcessor
        spp = SentencePieceProcessor(tokenzier_path)
        toks = [spp.bos_id()]
        start_pos = 0
        for _ in range(max_tokens):
            tok = model(Tensor([toks[start_pos:]]), 0 if start_pos == 0 else Variable("start_pos", 1, 1024).bind(start_pos), temperature).item()
            toks.append(tok)
            tart_pos += 1
            print(spp.decode(toks))

