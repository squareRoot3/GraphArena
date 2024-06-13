import concurrent.futures
from tasks import *
from openai import OpenAI
from collections import defaultdict
import networkx as nx
import random, os
import pickle
import walker
from collections import Counter
# Load the task and dataset

task = TSP_Task('dataset4')  # GED, MCS, TSP
task.load_dataset('hard')

def solve_problem(index):
    print(index)
    p = task.problem_set[index]
    task.problem_set[index]['approx_answer'] = task.approx_solver(p['graph'][0], p['graph'][1], timeout=30)
    task.problem_set[index]['greedy_answer'] = task.approx_solver(p['graph'][0], p['graph'][1], timeout=30)
    return index, task.problem_set[index]

# Use ProcessPoolExecutor to parallelize the loop
with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
    futures = [executor.submit(solve_problem, i) for i in range(len(task.problem_set))]
    for future in concurrent.futures.as_completed(futures):
        index, solved_problem = future.result()
        task.problem_set[index] = solved_problem

# Save the updated dataset
task.save_dataset('hard')