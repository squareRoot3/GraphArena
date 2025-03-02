import os
import json
from collections import defaultdict
from tasks import *
import numpy as np
from openai import OpenAI
import sys

client = OpenAI(
    base_url = "https://api.openai.com/v1/chat/completions", # Replace it if you use other API server.
    api_key = 'YOUR_API_KEY', # Replace the API key with your own
)

task_name = sys.argv[1]
difficulty = sys.argv[2]
task = globals()[task_name + '_Task']('dataset')
task.load_dataset(difficulty)

system_prompt = "You are an advanced AI specialized in solving graph problems. Provide the solution without writing or executing any code, and present your answer within brackets []. Do not use brackets in other places."
response_dict = {}

error_knt = 0
file_name = f"results/multi/{task_name}_{difficulty}.json"

if os.path.exists(file_name):
    with open(file_name, 'r') as f:
        response_dict = json.load(f)
        print(f"Continue")

for i in range(0, 500):
    if i in response_dict and len(response_dict[i]) > 10:
        print(i)
        continue
    response_dict[i] = []
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": task.insert_example(i, 1)}]
    # print(messages)
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model='gpt-4o-mini-2024-07-18',
            seed=42,
            temperature=1,
            n=16        
        )
        for j in range(16):
            response_dict[i].append(chat_completion.choices[j].message.content)
        print(i, response_dict[i])
    except Exception as e:
        print('Call API failed! ', e)
        error_knt += 1
        response_dict[i] = ['Error!']
    with open(file_name, 'w') as f:
        json.dump(response_dict, f)


