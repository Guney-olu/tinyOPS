# MLOPS using tinygrad

## Features
### AutoCNN 

```sh
python3 examples/TinyAutoML_CNN.py -e 1 dataset/train dataset/test
```

### LLaMA Inference 

Easy llama inference and quantization wiht fast-api autobuild support

```sh
python3 examples/lllama_chat.py --model models/llama-1b-hf-32768-fpf --size 1B --shard 1 --quantize nf4 --seed 42
```


