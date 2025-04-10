"""
The Llama model configurations and weight downloading utilities.

adopted from opt_config.py
"""

import dataclasses
import glob
import os
import numpy as np
from tqdm import tqdm

@dataclasses.dataclass(frozen=True)
class RopeConfig:
    rope_theta: float = 500000.0
    rope_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192
    head_dim: int = 128

@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str = "Llama-2-7b-hf"
    hf_token: str = ''
    hidden_act: str = "silu"
    input_dim: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    n_head: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    dtype: type = np.float16
    pad_token_id: int = 2
    vocab_size: int = 32000
    has_lm_head: bool = True
    rms_norm_eps: float = 1e-05
    rope_config: RopeConfig = RopeConfig()

    def model_bytes(self):
        h = self.input_dim
        intermediate = self.intermediate_size
        n_head = self.n_head
        head_dim = h // n_head
        return 2 * (self.vocab_size * h +
        self.num_hidden_layers * (
        # self-attention
        h * h + 2 * h * h / (self.n_head / self.num_key_value_heads) + h * h +
        # mlp
        3 * h * intermediate +
        # layer norm
        2 * h) +
        # head
        h + self.vocab_size * h)

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.input_dim


def get_llama_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[-1]

    if "-chat" in name:
        arch_name = name.replace("-chat", "")
    else:
        arch_name = name

    if arch_name == "llama-3.1-8b-instruct":
        config = LlamaConfig(name=name, hf_token=kwargs.get('hf_token'),
                             input_dim=4096, intermediate_size=14336, n_head=32,
                             num_hidden_layers=32, num_key_value_heads=8,
                             max_position_embeddings=131072, 
                             pad_token_id=128001, vocab_size=128256
                             )
    elif arch_name == "llama-3.2-3b-instruct":
        rope_config = RopeConfig(rope_factor=32.0)
        config = LlamaConfig(name=name, hf_token=kwargs.get('hf_token'),
                             input_dim=3072, intermediate_size=8192, n_head=24,
                             num_hidden_layers=28, num_key_value_heads=8,
                             max_position_embeddings=131072, 
                             pad_token_id=128001, vocab_size=128256,
                             has_lm_head=False, rope_config=rope_config
                             )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


def download_llama_weights(model_name, path, hf_token):
    import torch
    from safetensors import safe_open

    folder = "/mnt/" + model_name
    safetensors_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1]
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for safetensors_file in tqdm(safetensors_files, desc="Convert format"):
        with safe_open(safetensors_file, framework='pt') as stf:
            for name in tqdm(stf.keys(), leave=False):
                param = stf.get_tensor(name)
                name = name.replace("model.", "")
                param_path = os.path.join(path, name)
                with open(param_path, "wb") as f:
                    np.save(f, param.to(torch.float16).cpu().detach().numpy())
