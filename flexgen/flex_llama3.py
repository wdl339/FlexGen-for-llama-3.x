"""
Usage:
python3 -m flexgen.flex_llama --model meta-llama/Llama-2-7b-chat-hf --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""
import os
import torch
import argparse
from typing import Union
from transformers import AutoTokenizer
from flexgen.compression import CompressionConfig
from flexgen.llama3_config import LlamaConfig, get_llama_config, download_llama_weights, RopeConfig
from flexgen.pytorch_backend import Llama3TorchDevice, TorchDisk, TorchMixedDevice, fix_recursive_import
from flexgen.flex_opt import (Policy, init_weight_list, InputEmbed, OutputEmbed, SelfAttention, MLP,
                              TransformerLayer, OptLM, get_filename)
from flexgen.timer import timers
from flexgen.utils import (ExecutionEnv, GB, ValueHolder,
    array_1d, array_2d, str2bool, project_decode_latency)
from datetime import datetime

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

def get_test_inputs(prompt_len, num_prompts, tokenizer, prompt_file=None):
    if prompt_file:
        with open(prompt_file, "r") as f:
            prompt = f.read()
            prompts = [prompt]
    else:
        # prompts = ["My fault, my failure, is not in the passions I have, but in my lack of control of them. —Jack Kerouac The language of Friendship is not words, but meanings. —Henry David Thoreau Chapter One The party was a killer! Erin Albright paused a moment in the madness to take it all in. The slash and crash of music designed to get your ass moving had those asses crowding the dance floor. Lights shifting from steamy red to electric blue to hot pink made it all so frigging sexy! The bartender’s generous pours on tonight’s signature drink, Girl Power, didn’t hurt a thing. They’d chosen a Monday night at the Down and Dirty because they’d wanted the heat, the sexy, and an off night so they’d have plenty of room for the couple dozen friends they’d wanted to join in the celebration. Plus, Monday nights at the D&D meant holo-bands, so people could jump on the stage and join right in. And when they did, it added to the fun. Shauna jumped up onstage—again—and someone made the mistake of giving her a mic. Shauna had a voice like a cat in heat, and she used it to screech out the lyrics to “Bang Me Hard.” God, could she possibly be more adorable? And in five days, only five more days, Erin thought, on August 20, 2061, she’ll be my wife, and I’ll be hers. Together forever. At five-two, Shauna Hunnicut made Erin think of a sexy fairy, one with a wild tangle of red hair and big, beautiful blue eyes. And that smile? Another killer. The hair had caught her eye that first time, and the eyes had dazzled. But oh, that smile. It had simply done her in from the get-go. She’d walked into Fancy Feet for a pair of shoes, and walked out completely infatuated. She, the no-strings, live-life-for-today street artist had fallen, and hard, for the shoe store manager. Who’d have thought that fifteen months, three weeks, and two days later, they’d promise each other lifetimes? She couldn’t wait to make that promise, to hear Shauna make it to her. Shauna’s friend Becca—her friend, too, now—grabbed Erin’s hand. “Gotta shake it, baby!” She shook it with Becca on the dance floor, and like everyone else, joined in on the chorus. “Bang me, bang me harder. Oh! Bang me, bang me harder. Oh. Oh. Oh!” “This is so much fun!” Becca shouted, and shoved her swing of strawberry blond—now sweaty—hair back from her pretty face. “Why haven’t I ever been here before?” “Because it’s a sex club and you’re an upstanding young professional and executive at a stuffy Madison Avenue marketing firm?” “Junior executive at a stuffy Madison Avenue marketing firm.” Becca executed a spin. “Woo! And I might not be so upstanding after tonight! You and Shauna have to get married more often!” “One and done for me.” She looked back as Shauna wound up for the finish. “God, isn’t she cute? Is anybody more adorable than my soon-to-be wife?” “Loved her for years—in a straight-girl kind of way. I’m so happy for her. For you, too!” A little bit drunk, and sweaty with it, Becca wrapped her arms around Erin. Cheers erupted. Erin added her own as Shauna threw her hands in the air. “I’m going to go get my girl before she decides to do an encore.” Waving her own hands in the air, Erin wove her way through bodies to the stage. “Come down and dance with me, you sexy thing!” “Anytime, anywhere.” Face glowing, Shauna dropped down to her butt, then scooted the rest of the way off the stage. “This is so much fun!” In her tiny blue dress and mile-high heels, she wrapped around Erin. “You have the best ideas.” “My best idea ever was deciding to try on those wild pink shoes I saw in the window. Pink shoes led me to you. I love you, baby.” “I’m the luckiest woman in this club, in this city, possibly the world. Because I have you.” Swaying to the music, wrapped tight, they kissed. Soft, sweet, even as music boomed out a frantic beat. Who knew, Erin thought again. Who knew she’d find the woman of her dreams—dreams she hadn’t thought to dream? A woman who’d open her life to love, to plans, to the future. Everything before Shauna had been the now. Always just the right now, forget"]
        prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len, truncation=True).input_ids
    return (input_ids[0],) * num_prompts

class LlamaInputEmbed(InputEmbed):
    def __init__(self, config, env, policy):
        super().__init__(config, env, policy)

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst),))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 3
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), = weight_read_buf.pop()
        else:
            (w_token, _), = weight_read_buf.val

        h = self.compute.llama_input_embed(h, mask,
            w_token, self.config.pad_token_id, donate)
        hidden.val = h


class LlamaOutputEmbed(OutputEmbed):
    def __init__(self, config, env, policy):
        super().__init__(config, env, policy)

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        if self.config.has_lm_head:
            weight_specs = [
                # w_ln
                ((h,), dtype, path + "norm.weight"),
                # w_token
                ((v, h), dtype, path + "lm_head.weight"),
            ]
        else:
            weight_specs = [
                # w_ln
                ((h,), dtype, path + "norm.weight"),
                # w_token
                ((v, h), dtype, path + "embed_tokens.weight"),
            ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), w_token.smart_copy(dst1)))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.llama_output_embed(h, w_ln, w_token, self.config.rms_norm_eps, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h


class LlamaSelfAttention(SelfAttention):
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
            # w_k
            ((n_kv_head*head_dim, h), dtype, path + "self_attn.k_proj.weight"),
            # w_v
            ((n_kv_head*head_dim, h), dtype, path + "self_attn.v_proj.weight"),
            # w_o
            ((n_head*head_dim, h), dtype, path + "self_attn.o_proj.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_q, w_k, w_v, w_o = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_ln.smart_copy(dst2),
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1),
                w_v.smart_copy(dst1),
                w_o.smart_copy(dst1)))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        n_head = self.config.n_head
        n_kv_head = self.config.num_key_value_heads

        donate = [False] * 10
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_ln, donate[2]), (w_q, donate[3]), (w_k, donate[4]), (w_v, donate[5]),
             (w_o, donate[6])) = weight_read_buf.pop()
        else:
            ((w_ln, _), (w_q, _), (w_k, _), (w_v, _),
             (w_o, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            position_ids = torch.cumsum(mask.data, dim=1).int() * mask.data + 1
            if self.prefill_batch_size == 0:
                h, new_k_cache, new_v_cache = self.compute.llama_mha(h, position_ids, mask, w_ln,
                    w_q, w_k, w_v, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                    self.policy.compress_cache, self.policy.comp_cache_config)
            else:
                h, new_k_cache, new_v_cache = self.compute.llama_mha_batched(h, position_ids, mask, w_ln,
                    w_q, w_k, w_v, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                    self.policy.compress_cache, self.policy.comp_cache_config, batch_size=self.prefill_batch_size)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[8]), (v_cache, donate[9]) = cache_read_buf.pop()
            position_ids = torch.cumsum(mask.data, dim=1).int() * mask.data + 1
            position_ids = position_ids[:, -h.shape[1]].unsqueeze(1)
            h, new_k_cache, new_v_cache = self.compute.llama_mha_gen(h, position_ids, mask, w_ln,
                w_q, w_k, w_v, w_o, self.config.rms_norm_eps, n_head, n_kv_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class LlamaMLP(MLP):
    def __init__(self, config, env, policy, layer_id):
        super().__init__(config, env, policy, layer_id)

    def init_weight(self, weight_home, path):
        h, intermediate, dtype = (self.config.input_dim, self.config.intermediate_size, self.config.dtype)
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "post_attention_layernorm.weight"),
            # w_g
            ((intermediate, h), dtype, path + "mlp.gate_proj.weight"),
            # w_u
            ((intermediate, h), dtype, path + "mlp.up_proj.weight"),
            # w_d
            ((h, intermediate), dtype, path + "mlp.down_proj.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_g, w_u, w_d = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_ln.smart_copy(dst2),
                w_g.smart_copy(dst1),
                w_u.smart_copy(dst1),
                w_d.smart_copy(dst1)))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 5
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_ln, donate[1]), (w_g, donate[2]), (w_u, donate[3]),
             (w_d, donate[4])) = weight_read_buf.pop()
        else:
            ((w_ln, _), (w_g, _), (w_u, _), (w_d, _)) = weight_read_buf.val

        h = self.compute.llama_mlp(h, w_ln, w_g, w_u, w_d, self.config.rms_norm_eps, donate)
        hidden.val = h


class LlamaTransformerLayer(TransformerLayer):
    def __init__(self, config, env, policy, i, prefill_batch_size):
        self.attention = LlamaSelfAttention(config, env, policy, i, prefill_batch_size)
        self.mlp = LlamaMLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute


class LlamaLM(OptLM):
    def __init__(self,
                 config: Union[str, LlamaConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy,
                 prefill_batch_size: int = 0):
        if isinstance(config, str):
            config = get_llama_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(LlamaInputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(LlamaSelfAttention(self.config, self.env, self.policy, i, prefill_batch_size))
                layers.append(LlamaMLP(self.config, self.env, self.policy, i))
            else:
                layers.append(LlamaTransformerLayer(self.config, self.env, self.policy, i, prefill_batch_size))
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
            download_llama_weights(self.config.name, self.path, self.config.hf_token)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)


def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer, args.file)

    llama_config = get_llama_config(args.model, hf_token=args.hf_token, pad_token_id=tokenizer.eos_token_id)

    gpu = Llama3TorchDevice("cuda:0", rope_config=llama_config.rope_config)
    cpu = Llama3TorchDevice("cpu", rope_config=llama_config.rope_config)
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

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

    cache_size = llama_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = llama_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {llama_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = LlamaLM(llama_config, env, args.path, policy, args.prefill_batch_size)

    try:
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

    model_size = llama_config.model_bytes()
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
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
        help="The model name.")
    parser.add_argument("--hf-token", type=str,
        help="The huggingface token for accessing gated repo.")
    parser.add_argument("--path", type=str, default="~/llama_weights",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)
