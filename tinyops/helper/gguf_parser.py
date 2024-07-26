"""
Source -> https://github.com/99991/pygguf/blob/main/gguf.py
"""
import struct
import numpy as np
from tinyops.helper.helpers import DATA_TYPES,GGML_BLOCK_SIZES,GGML_ELEMENTS_PER_BLOCK,GGML_NAMES,GGML_DEQUANTIZE

def read_value(f, data_type):
    if data_type == DATA_TYPES["string"]:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    elif data_type == DATA_TYPES["uint32"]:
        return struct.unpack("<I", f.read(4))[0]

    elif data_type == DATA_TYPES["uint64"]:
        return struct.unpack("<Q", f.read(8))[0]

    elif data_type == DATA_TYPES["int64"]:
        return struct.unpack("<q", f.read(8))[0]

    elif data_type == DATA_TYPES["int32"]:
        return struct.unpack("<i", f.read(4))[0]

    elif data_type == DATA_TYPES["float32"]:
        return struct.unpack("<f", f.read(4))[0]

    elif data_type == DATA_TYPES["float64"]:
        return struct.unpack("<d", f.read(4))[0]

    elif data_type == DATA_TYPES["bool"]:
        return struct.unpack("<?", f.read(1))[0]

    elif data_type == DATA_TYPES["uint8"]:
        return struct.unpack("<B", f.read(1))[0]

    elif data_type == DATA_TYPES["int8"]:
        return struct.unpack("<b", f.read(1))[0]

    elif data_type == DATA_TYPES["uint16"]:
        return struct.unpack("<H", f.read(2))[0]

    elif data_type == DATA_TYPES["int16"]:
        return struct.unpack("<h", f.read(2))[0]

    elif data_type == DATA_TYPES["array"]:
        data_type, count = struct.unpack("<IQ", f.read(4 + 8))
        return [read_value(f, data_type) for _ in range(count)]

    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")

def load_gguf(f):
    f.seek(4) #file pointer to position correctly
    values = struct.unpack("<IQQ", f.read(4+8+8))
    _, n_tensors, n_kv = values
    info = {}
    for _ in range(n_kv):
        name = read_value(f, DATA_TYPES["string"])

        data_type = struct.unpack("<I", f.read(4))[0]

        info[name] = read_value(f, data_type)

    tensor = {}
    for _ in range(n_tensors):
        name = read_value(f,DATA_TYPES["string"])
        shape_len = read_value(f,DATA_TYPES["uint32"])
        shape = [read_value(f, DATA_TYPES["uint64"]) for _ in range(shape_len)]
        ggml_type = read_value(f, DATA_TYPES["uint32"])
        bad_offset = read_value(f, DATA_TYPES["uint64"])
        tensor[name] = {
        "ggml_type": ggml_type,
         "shape": shape,
        "bad_offset": bad_offset,
        }
    
    start = f.tell()
    for t in tensor.values():
        offset = start + t["bad_offset"]
        # Alignment is 32 by default.
        # https://github.com/ggerganov/ggml/blob/e1daebbf9d38d510ba456c4d50b4500a73ac2b14/docs/gguf.md?plain=1#L253
        alignment = info.get("general.alignment", 32)
        offset += (alignment - offset % alignment) % alignment

        t["offset"] = offset

    return info, tensor

def load_gguf_tensor(f,tensorinfo,name):
    t = tensorinfo[name]
    offset = t["offset"]
    shape = t["shape"]
    ggml_type = t["ggml_type"]
    
    if ggml_type not in GGML_NAMES:
        raise NotImplementedError(f"ggml_type {ggml_type} not implemented")
    
    ggml_name = GGML_NAMES[ggml_type]
    
    block_size = GGML_BLOCK_SIZES[ggml_name]
    elements_per_block = GGML_ELEMENTS_PER_BLOCK[ggml_name]
    dequantize = GGML_DEQUANTIZE[ggml_name]

    num_elements = np.prod(shape)

    f.seek(offset)

    size = num_elements * block_size // elements_per_block
    data = f.read(size)
    values = dequantize(data)

    return values.reshape(shape[::-1])

# GGUF way of stroring weights 
def translate_name(name):
    if name == "output.weight":
        return "lm_head.weight"

    if name == "token_embd.weight":
        return "model.embed_tokens.weight"

    if name == "output_norm.weight":
        return "model.norm.weight"

    name = name.replace("blk.", "model.layers.")
    name = name.replace(".attn_norm.weight", ".input_layernorm.weight")
    name = name.replace(".ffn_down.weight", ".mlp.down_proj.weight")
    name = name.replace(".ffn_gate.weight", ".mlp.gate_proj.weight")
    name = name.replace(".ffn_up.weight", ".mlp.up_proj.weight")
    name = name.replace(".ffn_norm.weight", ".post_attention_layernorm.weight")
    name = name.replace(".attn_q.weight", ".self_attn.q_proj.weight")
    name = name.replace(".attn_k.weight", ".self_attn.k_proj.weight")
    name = name.replace(".attn_v.weight", ".self_attn.v_proj.weight")
    name = name.replace(".attn_output.weight", ".self_attn.o_proj.weight")

    return name