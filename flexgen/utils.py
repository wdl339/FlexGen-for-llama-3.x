import argparse
import dataclasses
from attr import define, field
from attr.setters import frozen
import functools
import gc
import math
import os
from typing import Tuple, Union, Optional, Any, Sequence, List

import numpy as np
import torch


KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12

import psutil
import threading
import time

class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.max_memory = 0
        self.stop_flag = False
        self.interval = interval
        self.process = psutil.Process()
        
    def monitor(self):
        while not self.stop_flag:
            current_memory = self.process.memory_info().rss
            
            if current_memory > self.max_memory:
                self.max_memory = current_memory
                
            time.sleep(self.interval)
    
    def start(self):
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def get_cur_mem(self):
        return self.process.memory_info().rss
        
    def stop(self):
        self.stop_flag = True
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        return self.max_memory

@dataclasses.dataclass(frozen=True)
class Task:
    """A generation task."""
    inputs: Union[np.array, List[List[int]]]
    prompt_len: int
    gen_len: int
    cut_gen_len: Optional[int]

    do_sample: bool
    temperature: float
    stop: Optional[int]


@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """Hardware environment."""
    gpu: Any = None
    cpu: Any = None
    disk: Any = None
    mixed: Any = None
    clear_cache: bool = False

    @classmethod
    def create(cls, offload_dir):
        # fix recursive import
        from flexgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
        gpu = TorchDevice("cuda:0")
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir)
        return cls(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    def close_copy_threads(self):
        self.disk.close_copy_threads()


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark results."""
    prefill_latency: float
    prefill_throughput: float
    decode_latency: float
    decode_throughput: float
    total_latency: float
    total_throughput: float


np_dtype_to_torch_dtype = {
    np.float16: torch.float16, np.float32: torch.float32, np.uint8: torch.uint8,
    np.int8: torch.int8, np.int32: torch.int32, np.int64: torch.int64,
    bool: torch.bool,
}

torch_dtype_to_np_dtype = {
    torch.float16: np.float16, torch.float32: np.float32,
    torch.uint8: np.uint8, torch.int8: np.int8, torch.int32: np.int32,
    torch.int64: np.int64, torch.bool: bool,
}

torch_dtype_to_num_bytes = {
    torch.float16: 2, torch.float32: 4,
    torch.int8: 1, torch.uint8: 1, torch.int32: 4, torch.int64: 8,
    torch.bool: 1,
}

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


def piecewise_linear_func(xs, ys):
    """Return a function created by linear inerpolation."""
    indices = np.argsort(xs)
    xs = [xs[i] for i in indices]
    ys = [ys[i] for i in indices]

    # pad left and right
    k = 1e5
    delta_x_left = xs[0] - xs[1]
    delta_y_left = ys[0] - ys[1]
    delta_x_right = xs[-1] - xs[-2]
    delta_y_right = ys[-1] - ys[-2]

    xs = [xs[0] + delta_x_left * k] + xs + [xs[-1] + delta_x_right * k]
    ys = [ys[0] + delta_y_left * k] + ys + [ys[-1] + delta_y_right * k]

    return functools.partial(piecewise_linear_func_ret_func, xs, ys)


def piecewise_linear_func_ret_func(xs, ys, x):
    assert x >= xs[0] and x <= xs[-1]
    return np.interp(x, xs, ys)


def sample_from_range(n, k):
    assert n >= 1

    if k == -1:
        ret = [1]
        while ret[-1] * 2 < n:
            ret.append(ret[-1] * 2)
        return ret
    else:
        if k == 1: return [1]
        step = (n - 1) // (k - 1)
        return list(range(1, n + 1, step))


def cpu_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and not obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem


def torch_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        print(tensor.shape, tensor.data_ptr())

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem


class ValueHolder:
    def __init__(self):
        self.val = None

    def store(self, val):
        assert self.val is None
        self.val = val

    def pop(self):
        ret = self.val
        self.val = None
        return ret

    def clear(self):
        self.val = None


def array_1d(a, cls):
    return [cls() for _ in range(a)]


def array_2d(a, b, cls):
    return [[cls() for _ in range(b)] for _ in range(a)]


def array_3d(a, b, c, cls):
    return [[[cls() for _ in range(c)] for _ in range(b)] for _ in range(a)]


def array_4d(a, b, c, d, cls):
    return [[[[cls() for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]


def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[S, B, H]
        indices: Tensor[K, B]
    Returns:
        Tensor[K, B, H]
    """
    S, B, H = vectors.shape
    K, B2 = indices.shape
    assert B == B2
    indices = indices.reshape(K, B, 1).expand(K, B, H)
    out = vectors.gather(dim=0, index=indices)
    return out


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def project_decode_latency(costs, prompt_len, gen_len):
    decode_costs = costs[1:]

    if gen_len / prompt_len < 0.1:
        warmup = 2
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))
    else:
        warmup = 2
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))

        #assert len(decode_costs) >= 4
        #warmup = 2
        #xs = np.arange(warmup, len(decode_costs))
        #ys = np.asarray(decode_costs[warmup:])
        #curve = np.poly1d(np.polyfit(xs, ys, deg=1))
        #ys_pred = [curve(x) for x in range(gen_len-1)]
        #decode_latency = sum(ys_pred)

        #print([round(x, 4) for x in decode_costs])
        #print([round(x, 4) for x in ys_pred])

    return decode_latency


def write_benchmark_log(filename, model_size, cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput):

    log_str = (f"model size: {model_size/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (p): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t"
               f"projected: {projected}\n"
               f"prefill latency: {prefill_latency:.3f} s\t"
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"
               f"decode latency: {decode_latency:.3f} s\t"
               f"decode throughput: {decode_throughput:.3f} token/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} token/s")
    with open(filename, "a") as fout:
        fout.write(log_str + "\n")

    return log_str


def read_benchmark_log(filename):
    with open(filename) as fin:
        lines = fin.readlines()

    def extract(line):
        a, b = line.split("\t")
        latency = a[a.index(":") + 1:a.index(" s")]
        throughput = b[b.index(":") + 1:b.index(" to")]
        return float(latency), float(throughput)

    prefill_latency, prefill_throughput = extract(lines[2])
    decode_latency, decode_throughput = extract(lines[3])
    total_latency, total_throughput = extract(lines[4])

    return BenchmarkResult(
        prefill_latency, prefill_throughput,
        decode_latency, decode_throughput,
        total_latency, total_throughput,
    )
