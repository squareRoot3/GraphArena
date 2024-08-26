import os, sys
import multiprocessing as mp
from tasks import *
from openai import OpenAI
import networkx as nx
import random
import numpy as np
import argparse
import fast_tsp
import time
from functools import partial

def process_problem(problem, result_dict):
    exact_answer, path = task.exact_solver(*problem['graph'])
    print(f"Processed problem {problem['id']}")
    result_dict[problem['id']] = {'exact_answer': exact_answer, 'path': path}

def save_results(task, result_dict):
    for problem_id, result in result_dict.items():
        idx = next(i for i, prob in enumerate(task.problem_set) if prob['id'] == problem_id)
        task.problem_set[idx]['exact_answer'] = result['exact_answer']
        task.problem_set[idx]['path'] = result['path']
    task.save_dataset('hard')
    print("Results saved.")

if __name__ == '__main__':
    task = MCS_Task('dataset')
    task.load_dataset('hard')

    num_cores = mp.cpu_count()

    manager = mp.Manager()
    result_dict = manager.dict()

    pool = mp.Pool(processes=num_cores)

    for problem in task.problem_set[:500]:
        pool.apply_async(process_problem, args=(problem, result_dict))

    start_time = time.time()
    while pool._cache:
        time.sleep(10)  
        if time.time() - start_time >= 600:  
            save_results(task, result_dict)
            start_time = time.time()

    pool.close()
    pool.join()

    save_results(task, result_dict)

    print("Processing complete.")