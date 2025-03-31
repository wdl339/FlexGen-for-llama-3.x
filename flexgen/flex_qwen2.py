"""
Usage:
python3 -m flexgen.flex_qwen --model Qwen/Qwen1.5-0.5B-Chat --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""
import os
import torch
import argparse
from typing import Union
from transformers import AutoTokenizer
from flexgen.compression import CompressionConfig
from flexgen.qwen2_config import QwenConfig, get_qwen_config, download_qwen_weights
from flexgen.flex_llama import LlamaInputEmbed, LlamaOutputEmbed, LlamaMLP
from flexgen.pytorch_backend import TorchDisk, TorchMixedDevice
from flexgen.qwen_backend import QwenTorchDevice, fix_recursive_import
from flexgen.flex_opt import (Policy, init_weight_list, SelfAttention, TransformerLayer,
                              OptLM, get_filename)
from flexgen.timer import timers
from flexgen.utils import (ExecutionEnv, GB, ValueHolder, MemoryMonitor,
    array_1d, array_2d, str2bool, project_decode_latency, get_test_inputs)
from datetime import datetime

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


class QwenSelfAttention(SelfAttention):
    def __init__(self, config, env, policy, layer_id, prefill_batch_size):
        super().__init__(config, env, policy, layer_id)
        self.prefill_batch_size = prefill_batch_size

    def init_weight(self, weight_home, path):
        h, n_head, n_kv_head, dtype = (self.config.input_dim, self.config.n_head, self.config.num_key_value_heads, self.config.dtype)
        head_dim = h // n_head
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "input_layernorm.weight"),
            # w_q
            ((h, n_head*head_dim), dtype, path + "self_attn.q_proj.weight"),
            # b_q
            ((h,), dtype, path + "self_attn.q_proj.bias"),
            # w_k
            ((n_kv_head*head_dim, h), dtype, path + "self_attn.k_proj.weight"),
            # b_k
            ((n_kv_head*head_dim,), dtype, path + "self_attn.k_proj.bias"),
            # w_v
            ((n_kv_head*head_dim, h), dtype, path + "self_attn.v_proj.weight"),
            # b_v
            ((n_kv_head*head_dim,), dtype, path + "self_attn.v_proj.bias"),
            # w_o
            ((n_head*head_dim, h), dtype, path + "self_attn.o_proj.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_q, b_q, w_k, b_k, w_v, b_v, w_o = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_ln.smart_copy(dst2),
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_o.smart_copy(dst1)))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        n_head = self.config.n_head
        n_kv_head = self.config.num_key_value_heads

        donate = [False] * 12
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_ln, donate[2]), (w_q, donate[3]), (b_q, donate[4]), (w_k, donate[5]), (b_k, donate[6]),
             (w_v, donate[7]), (b_v, donate[8]), (w_o, donate[9])) = weight_read_buf.pop()
        else:
            ((w_ln, _), (w_q, _), (b_q, _), (w_k, _), (b_k, _), (w_v, _), (b_v, _),
             (w_o, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            position_ids = torch.cumsum(mask.data, dim=1).int() * mask.data + 1
            if self.prefill_batch_size == 0:
                h, new_k_cache, new_v_cache = self.compute.qwen_mha(h, position_ids, mask, w_ln,
                    w_q, b_q, w_k, b_k, w_v, b_v, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps, self.config.rope_theta,
                    self.policy.compress_cache, self.policy.comp_cache_config)
            else:
                h, new_k_cache, new_v_cache = self.compute.qwen_mha_batched(h, position_ids, mask, w_ln,
                    w_q, b_q, w_k, b_k, w_v, b_v, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps, self.config.rope_theta,
                    self.policy.compress_cache, self.policy.comp_cache_config, batch_size=self.prefill_batch_size)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[10]), (v_cache, donate[11]) = cache_read_buf.pop()
            position_ids = torch.cumsum(mask.data, dim=1).int() * mask.data + 1
            position_ids = position_ids[:, -h.shape[1]].unsqueeze(1)
            h, new_k_cache, new_v_cache = self.compute.qwen_mha_gen(h, position_ids, mask, w_ln,
                w_q, b_q, w_k, b_k, w_v, b_v, w_o, self.config.rms_norm_eps, self.config.rope_theta, n_head, n_kv_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class QwenTransformerLayer(TransformerLayer):
    def __init__(self, config, env, policy, i):
        self.attention = QwenSelfAttention(config, env, policy, i)
        self.mlp = LlamaMLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute


class QwenLM(OptLM):
    def __init__(self,
                 config: Union[str, QwenConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy,
                 prefill_batch_size: int = 0):
        if isinstance(config, str):
            config = get_qwen_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(LlamaInputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(QwenSelfAttention(self.config, self.env, self.policy, i, prefill_batch_size))
                layers.append(LlamaMLP(self.config, self.env, self.policy, i))
            else:
                layers.append(QwenTransformerLayer(self.config, self.env, self.policy, i, prefill_batch_size))
        layers.append(LlamaOutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_qwen_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)


def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer, args.file)

    qwen_config = get_qwen_config(args.model, pad_token_id=tokenizer.eos_token_id)

    gpu = QwenTorchDevice("cuda:0")
    cpu = QwenTorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]),
                      clear_cache=args.clear_cache)

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    cache_size = qwen_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = qwen_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {qwen_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")


    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    
    try:
        memory_before_init = memory_monitor.get_cur_mem()
        
        print("init weight...")
        model = QwenLM(qwen_config, env, args.path, policy, args.prefill_batch_size)
        
        memory_after_init = memory_monitor.get_cur_mem()

        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()
        max_memory = memory_monitor.stop()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        # for i in [0, len(outputs)-1]:
        show_str += f"{0}: {outputs[0]}\n"
        show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file_dir == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        filename = args.log_file_dir + "/" + filename

    model_size = qwen_config.model_bytes()
    prompt_len = len(inputs[0])
    eval_len = len(output_ids[0]) - prompt_len
    prefill_speed = num_prompts * prompt_len / prefill_latency
    decode_speed = num_prompts * eval_len / decode_latency
    
    all_content = outputs[0]
    new_tokens = output_ids[0][len(inputs[0]):] 
    generate_content_list = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    generate_content = ''.join(generate_content_list)

    log_str = (f"model size: {model_size/GB:.3f} GB\t"
                f"cache size: {cache_size/GB:.3f} GB\t"
                f"hidden size (p): {hidden_size/GB:.3f} GB\n"
                f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t"
                f"peak cpu mem: {cpu_peak_mem / GB:.3f} GB\n"
                "\n"
                f"whether to clear page cache: {args.clear_cache}\n"
                f"mem before init: {memory_before_init / GB:.3f} GB\t"
                f"mem after init: {memory_after_init / GB:.3f} GB\t"
                f"max mem used: {max_memory / GB:.3f} GB\n"
                "\n"
                f"prompt len: {prompt_len}\n"
                f"eval len: {eval_len}\n"
                f"prefill speed: {prefill_speed:.3f} tokens/s\n"
                f"eval speed: {decode_speed:.3f} tokens/s\n"
                "\n"
                f"prefill latency: {prefill_latency:.3f} s\t"
                f"prefill throughput: {prefill_throughput:.3f} tokens/s\n"
                f"eval latency: {decode_latency:.3f} s\t"
                f"eval throughput: {decode_throughput:.3f} tokens/s\n"
                f"total latency: {total_latency:.3f} s\t"
                f"total throughput: {total_throughput:.3f} tokens/s\n"
                "\n"
                f"generate content: {generate_content}\n\n"
                f"all content: {all_content}\n"
            )
    with open(filename, "a") as fout:
        fout.write(log_str + "\n")

    if args.verbose >= 1:
        print(log_str)


def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B-Chat",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/qwen_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--file", type=str, default="",
        help="prompt file")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--prefill-batch-size", type=int, default=0)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")
    parser.add_argument("--log-file-dir", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--clear-cache", action="store_true",
        help="Whether to clear page cache after each iteration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)
