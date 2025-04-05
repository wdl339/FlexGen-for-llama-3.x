import requests

url = "http://127.0.0.1:8080/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "model",
    "messages": [
        {
            "content": "Paris is the capital city of",
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