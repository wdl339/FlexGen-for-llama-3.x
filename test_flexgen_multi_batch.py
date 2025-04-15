import os
import json
import subprocess
from datetime import datetime

###
# You should use sudo permissions to run this script (because of "--clear-cache")
# sudo -E python3 /mnt/FlexGen/test_flexgen_multi_batch.py
###

def run_llama_cli(file_path, prompt_len, output_path, csv_file, offload_to_disk
                  , prefill_batch_size=512, num_gpu_batches=1):
    env = os.environ.copy()

    project = "flexgen.flex_llama3"
    weight_path = "/mnt/llama_weights"
    if model == "qwen-2.5-7b-instruct":
        project = "flexgen.flex_qwen2"
        weight_path = "/mnt/qwen_weights"
    
    cpu_percent = 100
    if offload_to_disk:
        cpu_percent = 0
    
    command = [
        "python3", 
        "-m", project,
        "--model", f"/mnt/{model}", 
        "--path", weight_path,
        "--offload-dir", "/mnt/FlexGen/offload_dir", 
        "--file", file_path,
        "--prompt-len", str(prompt_len),
        "--gen-len", "512",
        "--gpu-batch-size", "1",
        "--num-gpu-batches", str(num_gpu_batches),
        "--prefill-batch-size", str(prefill_batch_size),
        "--percent", "100", "0", "0", str(cpu_percent), "100", "0", 
        "--attn-sparsity", "0.1",
        "--log-file-dir", output_path,
        "--compress-weight",
        "--csv-file", csv_file,
    ]
    
    if offload_to_disk:
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

for model in [
    "llama-3.1-8b-instruct", 
    # "llama-3.2-3b-instruct",
    "qwen-2.5-7b-instruct"
    ]:
    offload_disk = True
    for batch_size in [1, 2, 4, 8]:
        for length in lengths:
            input_folder = '/mnt/LongBench-v2/extracted_contexts/'
            output_folder = f'/mnt/FlexGen/aaa_test_output_multi_batch/{model}/'
            length_folder = os.path.join(input_folder, f"context_{length}")
            test_output_folder = os.path.join(output_folder, f"batch_{batch_size}")
            test_output_folder = os.path.join(test_output_folder, f"context_{length}")
            if not os.path.exists(test_output_folder):
                os.makedirs(test_output_folder)

            test_case_num = 3
            prefill_batch_size = 512

            if length == 64000 or length == 128000:
                test_case_num = 2
                
            if model == "llama-3.1-8b-instruct" or model == "qwen-2.5-7b-instruct":
                if length >= 16000:
                    prefill_batch_size = 64

            count = 0
            for filename in os.listdir(length_folder):
                if filename.endswith('.txt'):
                    count += 1
                    file_path = os.path.join(length_folder, filename)
                    csv_file = os.path.join(test_output_folder, f"sync_time_{length}_{count}.csv")
                    res = run_llama_cli(file_path, length, test_output_folder, csv_file,
                                        offload_disk, prefill_batch_size, batch_size)
                    # print(res)
                    if count >= test_case_num:
                        break