from transformers import AutoTokenizer
from flexgen.compression import CompressionConfig
from flexgen.llama3_config import get_llama_config
from flexgen.pytorch_backend import Llama3TorchDevice, TorchDisk, TorchMixedDevice, fix_recursive_import
from flexgen.flex_opt import Policy
from flexgen.timer import timers
from flexgen.utils import ExecutionEnv
from flexgen.flex_llama3 import (get_test_inputs, LlamaLM)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
import uvicorn
import threading

fix_recursive_import()

MODEL = "/disk2/wdl/llama-3.2-3b-instruct"
PATH = "/disk2/wdl/FlexGen/llama_weights"
OFFLOAD_DIR = "/disk2/wdl/FlexGen/offload_dir"
CUT_GEN_LEN = None
DEBUG_MODE = None
GPU_BATCH_SIZE = 1
NUM_GPU_BATCHES = 1
PREFILL_BATCH_SIZE = 512
PERCENT = [100, 0, 100, 0, 100, 0]
SEP_LAYER = True
PIN_WEIGHT = True
CPU_CACHE_COMPUTE = False
ATTN_SPARSITY = 0.1
COMPRESS_WEIGHT = False
COMPRESS_CACHE = False
VERBOSE = 2
OVERLAP = True

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id
num_prompts = NUM_GPU_BATCHES * GPU_BATCH_SIZE
cut_gen_len = CUT_GEN_LEN

llama_config = get_llama_config(MODEL, pad_token_id=tokenizer.eos_token_id)

gpu = Llama3TorchDevice("cuda:0", rope_config=llama_config.rope_config)
cpu = Llama3TorchDevice("cpu", rope_config=llama_config.rope_config)
disk = TorchDisk(OFFLOAD_DIR)
env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]),
                      clear_cache=False)

policy = Policy(GPU_BATCH_SIZE, NUM_GPU_BATCHES,
                    PERCENT[0], PERCENT[1],
                    PERCENT[2], PERCENT[3],
                    PERCENT[4], PERCENT[5],
                    OVERLAP, SEP_LAYER, PIN_WEIGHT,
                    CPU_CACHE_COMPUTE, ATTN_SPARSITY,
                    COMPRESS_WEIGHT,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    COMPRESS_CACHE,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))

print("init weight...")
model = LlamaLM(llama_config, env, PATH, policy, PREFILL_BATCH_SIZE)

warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
print("warmup...")
model.generate(warmup_inputs, max_new_tokens=1, verbose=VERBOSE)

def run_flexgen(chat, max_tokens):
    print("run_flexgen...")
    message = tokenizer.apply_chat_template(chat, tokenize=False)
    inputs = [message]
    input_ids = tokenizer(inputs, add_special_tokens=False).input_ids
    timers("generate").reset()
    output_ids = model.generate(
        (input_ids[0],) * num_prompts,
        max_new_tokens=max_tokens,
        debug_mode=DEBUG_MODE, 
        cut_gen_len=cut_gen_len, 
        verbose=VERBOSE
    )
    print("finish generate")

    result = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    return result.replace(message,"")\
                    .replace("<|start_header_id|>assistant<|end_header_id|>\n\n","")\
                    .replace("<|eot_id|>","")

app = FastAPI()
lock = threading.Lock()

class CompletionRequest(BaseModel):
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    stop: Optional[str] = None
    temperature: Optional[float] = None
    prompt: Optional[str] = ""

    # "messages": [
    #     {
    #         "content": "...",
    #         "role": "user"
    #     }
    # ],

class CompletionResponseChoice(BaseModel):
    message: Dict[str, str]

class CompletionResponse(BaseModel):
    choices: List[CompletionResponseChoice]

@app.post("/v1/chat/completions")
async def create_completion(request: CompletionRequest):
    try:
        completion = run_flexgen(
            request.messages,
            request.max_tokens
        )

        return CompletionResponse(
            choices=[
                CompletionResponseChoice(
                    message={"content": completion}
                )
            ]
        )
    
    except Exception as e:
        lock.release()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)