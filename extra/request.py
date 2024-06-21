import requests
import json

url = 'http://localhost:7776/v1/chat/completions'
headers = {'Content-Type': 'application/json'}
data = {
    "stream": True,
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ]
}

# Send the POST request and handle the streaming response
with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(json.loads(decoded_line[5:]))  # Strip "data: " from the beginning
    else:
        print(f"Request failed with status code: {response.status_code}")

