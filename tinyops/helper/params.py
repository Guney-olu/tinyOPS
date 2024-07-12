from sentencepiece import SentencePieceProcessor
import tiktoken, sys
from tiktoken.load import load_tiktoken_bpe

class TikToken:
  num_reserved_special_tokens: int = 256
  pat_str: str =  r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

  def __init__(self, model_file):
    mergeable_ranks = load_tiktoken_bpe(model_file)
    self.num_base_tokens = len(mergeable_ranks)

    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
      ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, self.num_reserved_special_tokens - 5)
      ]
    self.special_tokens = {
        token: self.num_base_tokens + i for i, token in enumerate(special_tokens)
    }

    self.model = tiktoken.Encoding(
      name=model_file,
      pat_str=self.pat_str,
      mergeable_ranks=mergeable_ranks,
      special_tokens=self.special_tokens,
    )

  def decode(self, toks): return self.model.decode([t for t in toks if t < self.num_base_tokens])
  def encode(self, s): return self.model.encode(s)

  def bos_id(self): return self.special_tokens["<|begin_of_text|>"]
  def eos_id(self): return self.special_tokens["<|end_of_text|>"]
  def vocab_size(self): return self.model.n_vocab

#LLama Model support
#TODO  ADD Mistral and Gemma
MODEL_PARAMS = {
  "1": {
    "7B": {
      "args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 32,"n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 11008},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "n_heads": 40, "n_kv_heads": 40,"n_layers": 40, "norm_eps": 1e-06, "vocab_size": 32000, "hidden_dim": 13824},
      "files": 2,
    },
    "30B": {
      "args": {"dim": 6656, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-06, "vocab_size": 32000, "hidden_dim": 17920},
      "files": 4,
    },
    "65B": {
      "args": {"dim": 8192, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 22016},
      "files": 8,
    },
    "tokenizer": SentencePieceProcessor,
  },
  "2": {
    "7B": {
      "args": {"dim": 4096, "n_heads": 32,"n_kv_heads": 32,"n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 11008},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "n_heads": 40,"n_kv_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 13824},
      "files": 2,
    },
    "70B": {
      "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 28672},
      "files": 8,
    },
    "tokenizer": SentencePieceProcessor,
  },
  "3": {
    "8B": {
      "args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-05, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 14336},
      "files": 1,
    },
    "8B-Chat": {
      "args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-05, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 14336},
      "files": 1,
    },
    "70B": {
      "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 28672},
      "files": 8,
    },
    "70B-Chat": {
      "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 28672},
      "files": 8,
    },
    "tokenizer": TikToken,
  },
  "gemma": {
    "2B": {
      "args": {"dim": 2048, "n_heads": 8,"n_kv_heads": 1,"n_layers": 18, "norm_eps": 1e-05, "vocab_size": 256000, "hidden_dim": 16384},
      "files": 1,
    },
    "tokenizer": SentencePieceProcessor,
  }
}
