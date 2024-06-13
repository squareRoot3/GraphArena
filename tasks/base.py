import random
import pickle
import re
import signal
import functools
import pandas as pd
import time
import networkx as nx
import numpy as np
import os
from rdkit import Chem
from matplotlib import pyplot as plt


def sample_node_size(min_nodes, max_nodes):
    exp = (max_nodes - min_nodes)/2
    while True:
        nodes_num = int(np.random.exponential(exp))
        if nodes_num >= min_nodes and nodes_num <= max_nodes:
            return nodes_num

def find_node_by_name(graph, name):
    for node, data in graph.nodes(data=True):
        if data.get('name') == name:
            return node
    return None

def smiles_to_networkx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=bond.GetBondType())
    return G

class NPTask(object):  # todo NPTask -> GraphTask
    def __init__(self, data_loc='./dataset', task_name='TSP'):
        self.data_loc = data_loc
        self.task_name = task_name
        self.problem_set = []
        
    def generate_dataset(self):
        raise NotImplementedError

    def check_solution(self):
        raise NotImplementedError
    
    def save_dataset(self, difficulty=None):
        if difficulty:
            pickle.dump(self.problem_set, open(f'{self.data_loc}/{self.task_name}_{difficulty}.pkl', 'wb'))
        else:
            pickle.dump(self.problem_set, open(f'{self.data_loc}/{self.task_name}.pkl', 'wb'))
        if len(self.examples) > 0:
            pickle.dump(self.examples, open(f'{self.data_loc}/{self.task_name}_examples.pkl', 'wb'))
            print(f'{len(self.examples)} examples generated.')

        print(f'{len(self.problem_set)} problems generated.')

    def load_dataset(self, difficulty=None, example=True):
        if difficulty:
            self.problem_set = pickle.load(open(f'{self.data_loc}/{self.task_name}_{difficulty}.pkl', 'rb'))
        else:
            self.problem_set = pickle.load(open(f'{self.data_loc}/{self.task_name}.pkl', 'rb'))
        if example:
            self.examples = pickle.load(open(f'{self.data_loc}/{self.task_name}_examples.pkl', 'rb'))
            
    def insert_example(self, id, num=2):
        prompt = self.problem_set[id]['problem_text']

        # selected_examples = random.sample(self.examples, num)
        marker_position = prompt.find('\n**Problem to Solve**')
        for i in range(num):
        # for i, example in enumerate(selected_examples):
            prompt = prompt[:marker_position] + f'\n**Example {num-i}**\n\n' + self.examples[(id+i)%100] + '\n' + prompt[marker_position:]
        return prompt
