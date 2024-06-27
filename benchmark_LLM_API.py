from tasks import *
from openai import OpenAI
from collections import defaultdict
import pickle
import argparse
import json
import datetime
from time import sleep
import os

llm_to_api = {
    "gpt4": "gpt-4o",
    "gpt": "gpt-3.5-turbo-0125", 
    "claude": "claude-3-haiku-20240307",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek": "deepseek-chat",
    "llama8b": "meta-llama/Llama-3-8b-chat-hf",
    "llama": "meta-llama/Llama-3-70b-chat-hf",
    "qwen7b": "qwen1.5-7b-chat",
    "qwen": "qwen1.5-72b-chat",
    "gemma": "gemma-7b-it",
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt', help='llm model name')
    parser.add_argument('--task', type=str, default='TSP', help='task name')
    parser.add_argument('--problem_num', type=int, default=10, help='number of problems')
    parser.add_argument('--example_num', type=int, default=2, help='number of examples')
    parser.add_argument('--difficulty', type=str, default='easy', help='problem difficulty')
    parser.add_argument('--resume', type=bool, default=False, help='resume from last checkpoint')
    parser.add_argument('--results', type=str, default='tmp', help='results location')
    parser.add_argument('--sleep', type=int, default=5, help='sleep seconds between API calls')

    args = parser.parse_args()
    classname = args.task + '_Task'
    task = globals()[classname]('dataset4')
    task.load_dataset(args.difficulty)
    error_knt = 0
    
    response_dict = defaultdict(dict)
    
    for llm in args.llm.split('-'):
        if 'gpt' in llm:
            client = OpenAI(
                base_url = "https://api.openai.com/v1/chat/completions", # Replace it to other API endpoint (e.g., AIMLAPI) if OpenAI is unavailable.
                api_key = 'YOUR_API_KEY', # Replace the API key with your own
            )
        elif 'deepseek' in llm:
            client = OpenAI(
                base_url = "https://api.deepseek.com",
                api_key = 'YOUR_API_KEY'
            )
        elif 'qwen' in llm:
            client = OpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="YOUR_API_KEY",
            )
        else:
            client = OpenAI(
                base_url = "https://api.aimlapi.com/",
                api_key = 'YOUR_API_KEY'
            )

        if args.resume and os.path.exists(f"results/tmp_{args.results}/{args.llm}_{args.task}_{args.difficulty}.json"):
            with open(f"results/tmp_{args.results}/{args.llm}_{args.task}_{args.difficulty}.json", 'r') as f:
                response_dict = json.load(f)
                print(f"Continue")

        if not os.path.exists(f"results/tmp_{args.results}"):
            os.makedirs(f"results/tmp_{args.results}")
        for i in range(0, args.problem_num):
            system_prompt = "You are an advanced AI specialized in solving graph problems. Provide the solution without writing or executing any code, and present your answer within brackets []. Do not use brackets in other places."
            i = str(i)
            if args.resume and i in response_dict and llm in response_dict[i] and response_dict[i][llm]:
                if response_dict[i][llm] != 'Error!':
                    print(i)
                    continue
            response_dict[i] = {}
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task.insert_example(int(i), args.example_num)},
                    ],
                    model=llm_to_api[llm],
                    seed=42,
                    temperature=0.1
                )
                response_dict[i][llm] = chat_completion.choices[0].message.content
                print(llm, i, response_dict[i][llm])
            except Exception as e:
                print('Call API failed! ', e)
                sleep(1)
                error_knt += 1
                response_dict[i][llm] = 'Error!'
            with open(f"results/tmp_{args.results}/{args.llm}_{args.task}_{args.difficulty}.json", 'w') as f:
                json.dump(response_dict, f)
            sleep(args.sleep)
    print('error_knt:', error_knt) # if error_knt > 0, please check the API key and endpoint and run again. The script will continue from prevoiusly failed samples.
    now = datetime.datetime.now()
    if not os.path.exists(f"results/{args.results}"):
        os.makedirs(f"results/{args.results}")
    with open(f"results/{args.results}/{args.llm}_{args.task}_{args.difficulty}_{now.strftime('%d_%H-%M')}.json", 'w') as f:
        json.dump(response_dict, f)
        