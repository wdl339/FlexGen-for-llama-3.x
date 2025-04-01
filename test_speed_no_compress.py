import os
import json
import subprocess
from datetime import datetime

###
# python3 /mnt/FlexGen/test_speed_no_compress.py
###

# model = "qwen-2.5-7b-instruct"
# model = "llama-3.1-8b-instruct"
model = "llama-3.2-3b-instruct"
input_folder = '/mnt/LongBench-v2/extracted_contexts/'
output_folder = f'/mnt/FlexGen/bbb_test_output_cuda/{model}/'

def run_llama_cli(file_path, prompt_len, output_path):
    env = os.environ.copy()

    project = "flexgen.flex_llama3"
    weight_path = "/mnt/llama_weights"
    if model == "qwen-2.5-7b-instruct":
        project = "flexgen.flex_qwen2"
        weight_path = "/mnt/qwen_weights"
    
    command = [
        "python3", 
        "-m", project,
        "--model", f"/mnt/{model}", 
        "--path", weight_path,
        # "--offload-dir", "/mnt/FlexGen/offload_dir", 
        "--file", file_path,
        "--prompt-len", str(prompt_len),
        "--gen-len", "512",
        "--gpu-batch-size", "1",
        "--prefill-batch-size", "512",
        # "--percent", "100", "0", "100", "0", "100", "0", 
        # "--attn-sparsity", "0.1",
        "--log-file-dir", output_path,
        # "--compress-weight"
    ]

    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    # print(result)
    return result

lengths = [\
            1000, \
            # 2000, \
            # 4000, \
            # 8000, \
            # 16000, \
            # 32000, \
            # 64000, \
            # 128000, \
]

for length in lengths:
    length_folder = os.path.join(input_folder, f"context_{length}")
    test_output_folder = os.path.join(output_folder, f"context_{length}")
    if not os.path.exists(test_output_folder):
        os.makedirs(test_output_folder)

    test_case_num = 3

    if length == 64000 or length == 128000:
        test_case_num = 2

    count = 0
    for filename in os.listdir(length_folder):
        if filename.endswith('.txt'):
            count += 1
            file_path = os.path.join(length_folder, filename)
            res = run_llama_cli(file_path, length, test_output_folder)
            # print(res)
            if count >= test_case_num:
                break