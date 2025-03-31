"""
The Qwen model configurations and weight downloading utilities.

adopted from opt_config.py
"""

import dataclasses
import glob
import os
import numpy as np
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class QwenConfig:
    name: str = "Qwen1.5-7B"
    hidden_act: str = "silu"
    input_dim: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    n_head: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    dtype: type = np.float16
    pad_token_id: int = 151643
    vocab_size: int = 151936

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
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim / (self.n_head / self.num_key_value_heads) * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_qwen_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[-1]

    if "-Chat" in name:
        arch_name = name.replace("-Chat", "")
    else:
        arch_name = name

    if arch_name == "qwen-2.5-7b-instruct":
        config = QwenConfig(name=name,
                            input_dim=3584, intermediate_size=18944, 
                            max_position_embeddings=32768, n_head=28, 
                            num_hidden_layers=28, num_key_value_heads=4, 
                            rms_norm_eps=1e-6, rope_theta=1000000.0, vocab_size=152064
                            )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


def download_qwen_weights(model_name, path):
    import torch
    from safetensors import safe_open

    folder = "/mnt/" + model_name
    safetensors_files = glob.glob(os.path.join(folder, "*.safetensors"))
    print(f"Found {len(safetensors_files)} files in {folder}")

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
