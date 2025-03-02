import json
import os
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

llms = {
    "graphwiz": "GraphWiz/LLaMA2-7B-DPO",
    "llama8b": "meta-llama/Meta-Llama-3-8B-Instruct"
}

def generate_answer(problem_text, tokenizer, model, args):
    input_ids_w_attnmask = tokenizer(
        problem_text,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids=input_ids_w_attnmask.input_ids,
        attention_mask=input_ids_w_attnmask.attention_mask,
        generation_config=GenerationConfig(
            max_new_tokens=args.max_token,
            do_sample=True,
            temperature=args.temp,  # t=0.0 raise error if do_sample=True
        ),
    ).tolist()
        
    real_output_ids = [
        output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)
    ]
    output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
    return output_strs[0]

def get_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(llms[model_name], padding_side="left")
    print(tokenizer.pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        llms[model_name],
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()
    return tokenizer, model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='******LLM inference on GraphArena******')
    parser.add_argument("--llm", type=str, default="graphwiz", help="LLM used for inference")
    parser.add_argument("--temp", type=float, default="0.1", help="Temperature of LLM")
    parser.add_argument("--max_token", type=int, default=2048, help="Max output token of LLM")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--problem_num", type=int, default=500, help="Number of problems to evaluate")
    
    args = parser.parse_args()

    tokenizer, model = get_tokenizer_and_model(args.llm)
    
    # inference
    all_data = json.load(open('dataset/GraphArena_text.json'))  # alternative: GraphArena_text_0shot.json
    results = 'results/'

    response_dict = {}
    for difficulty in ['easy', 'hard']:
        for task_name in ['Connected', 'Diameter', 'Distance', 'Neighbor', 'GED', 'MCP', 'MCS', 'MIS', 'MVC', 'TSP']:
            response_dict[task_name] = {'easy':{}, 'hard':{}}
            for i in range(args.problem_num):
                problem_text = all_data[task_name][difficulty][i]
                response_dict[task_name][difficulty][i] = {}
                response_dict[task_name][difficulty][i][args.llm] = generate_answer(problem_text, tokenizer, model, args)
        json.dump(response_dict[task_name][difficulty], open(f'{args.results}/{args.llm}_{task_name}_{difficulty}.json', 'w'))
