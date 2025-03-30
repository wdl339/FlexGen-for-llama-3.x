import requests

url = "http://127.0.0.1:8080/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "C:\\Users\\LenovoTest\\Desktop\\models\\llama-3.1.q4.moe.0122.fake.powerinfer.gguf",
    "messages": [
        {
            "content": "There is a single choice question about clinical knowledge. Answer the question by replying A, B, C or D.\nQuestion: What size of cannula would you use in a patient who needed a rapid blood transfusion (as of 2020 medical knowledge)?\nA. 18 gauge.\nB. 20 gauge.\nC. 22 gauge.\nD. 24 gauge.\nAnswer:",
            "role": "user"
        }
    ],
    "max_tokens": 100,
    "n": 1,
    "logprobs": False,
    "top_logprobs": None,
    "stop": None,
    "temperature": 0.7,
    "prompt": ""
}

response = requests.post(url, headers=headers, json=data)
print(response.json())