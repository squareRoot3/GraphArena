from tasks import *
from openai import OpenAI
import networkx as nx
import random
import os
import argparse


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='TSP', help='task name')
    parser.add_argument('--problem_num', type=int, default=500, help='number of problems')
    parser.add_argument('--loc', type=str, default='dataset', help='dataset location')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    classname = args.task + '_Task'
    set_seed(args.seed)
    task = globals()[classname](args.loc)
    task.generate_dataset(count=args.problem_num)
    for p in task.problem_set:
        print(len(p['problem_text']))
