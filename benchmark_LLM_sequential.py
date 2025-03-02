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
file_name = f"results/concat/{task_name}_{difficulty}.json"

if os.path.exists(file_name):
    with open(file_name, 'r') as f:
        response_dict = json.load(f)
        print(f"Continue")

for i in range(0, 500):
    if i in response_dict and len(response_dict[i]) > 3:
        print(i)
        continue
    response_dict[i] = []
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": task.insert_example(i, 1)}]

    for j in range(5):
        chat_completion = client.chat.completions.create(
            messages=messages,
            model='gpt-4o-mini-2024-07-18',
            seed=42,
            temperature=1,
        )
        content = chat_completion.choices[0].message.content
        response_dict[i].append(content)
        score = task.check_solution(i, content)
        if score == task.problem_set[i]['exact_answer']:
            break
        elif score == -1:
            response = 'The answer is missing. Please try again.'
        elif score == -2:
            response = 'The answer is hallucinary. Please carefully read the graph in the problem and try again.'
        else:
            response = f'The answer is suboptimal with a score of {score}. Please try again to find an optimal answer.'
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": response})

    print(i, response_dict[i])
    with open(file_name, 'w') as f:
        json.dump(response_dict, f)


