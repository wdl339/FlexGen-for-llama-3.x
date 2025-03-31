import os
import json
import subprocess
from datetime import datetime

###
# You should use sudo permissions to run this script (because of "--clear-cache")
# sudo -E python3 /mnt/FlexGen/test_flexgen_speed.py
###

model = "llama-3.1-8b-instruct"
# model = "llama-3.2-3b-instruct"
offload_disk = False
# offload_disk = True
input_folder = '/mnt/LongBench-v2/extracted_contexts/'
output_folder = f'/mnt/FlexGen/aaa_test_output_cuda/{model}/'

def run_llama_cli(file_path, prompt_len, output_path):
    env = os.environ.copy()

    cpu_percent = 100
    if offload_disk:
        cpu_percent = 0
    
    command = [
        "python3", 
        "-m", "flexgen.flex_llama3",
        "--model", f"/mnt/{model}", 
        "--path", "/mnt/llama_weights", 
        "--offload-dir", "/mnt/FlexGen/offload_dir", 
        "--file", file_path,
        "--prompt-len", str(prompt_len),
        "--gen-len", "512",
        "--gpu-batch-size", "1",
        "--prefill-batch-size", "512",
        "--percent", "100", "0", "0", str(cpu_percent), "100", "0", 
        "--attn-sparsity", "0.1",
        "--log-file-dir", output_path,
        "--compress-weight"
    ]
    
    if offload_disk:
        command += ["--clear-cache"]

    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    # print(result)
    return result

lengths = [\
            1000, \
            2000, \
            4000, \
            8000, \
            16000, \
            32000, \
            64000, \
            128000, \
]

for length in lengths:
    length_folder = os.path.join(input_folder, f"context_{length}")
    if offload_disk:
        test_output_folder = os.path.join(output_folder, f"offload_to_disk")
    else:
        test_output_folder = os.path.join(output_folder, f"offload_to_cpu")
    test_output_folder = os.path.join(test_output_folder, f"context_{length}")
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