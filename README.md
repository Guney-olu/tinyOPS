# MLOPS using tinygrad

## Features

### LLM Inference 
Easy LLM inference with quantization using tinyops using tinygrad as backened

```py
from tinyops import LLaMa,Gemma
from tinygrad import Device
from pathlib import Path

model_path = Path("Path to model")
tokenizer_path = Path("Path to tokenzier.model file")
shard = 1 # Num of devices
device = tuple(f"{Device.DEFAULT}:{i}" for i in range(shard)) if shard > 1 else Device.DEFAULT

#Tinyops ->  gen = [1,2,3] size = [7B,13B,70B] , quant = [nf4,int8]
llama = LLaMa.build(model_path, tokenizer_path, model_gen="1", model_size="7B", quantize="nf4", device=device)
output = LLaMa.generate(llama,5,"I am batman",0.7,device)
print(output)

#Gemma Support
gemm = Gemma.build(model_path, tokenizer_path, model_gen="gemma", model_size="2B",device=device)
output = Gemma.Benchmark(gemm,10,0.7,device)
#output = Gemma.generate(gemm,5,"I am batman",0.7,device)
```

### Model Support

| Model   | Completion | 8bit | 4bit | 2bit |
|---------|------------|------|------|------|
| llama-2 | ✅         | ✅   |  ✅   | -    |
| llama-3 | ✅         | ✅   |  ✅   | -    |
| gemma   | ✅         | -    |  -   | -    |
| mixtral | ✅         | -    |  -   | -    |

### AutoCNN 

```sh
python3 examples/TinyAutoML_CNN.py -e 1 dataset/train dataset/test
```



