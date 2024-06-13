from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
import networkx as nx
from collections import Counter
import random
import pickle
import re
import signal
import functools
import json
import numpy as np
import walker
import math
import time
import pandas as pd
#from base import *
from tasks.base import *
from tqdm import tqdm

class Distance_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(Distance_Task, self).__init__(data_loc, 'Distance')
        self.examples=[]


    def check_solution(self, problem_id, response):
        g = self.problem_set[problem_id]['graph']
        pattern = re.compile(r'\[(.*?)\]')
        matches = pattern.findall(response)
                    
        if matches:
            for match in reversed(matches):
                node_list = []
                match = match.split(",")
                name_list = [name.strip() for name in match]
                for name in name_list:
                    node = find_node_by_name(g, name)
                    if node is None:
                        #print('Node not found:', name)
                        # return -2
                        continue
                    node_list.append(node)
                if len(node_list) == 0:
                    continue
                if self.is_feasible(g, node_list, problem_id):
                    return len(node_list)-1
            return -2
        return -1
    
    def is_feasible(self, g, route_list, problem_id):
        source = self.problem_set[problem_id]['source']
        target = self.problem_set[problem_id]['target']
        if route_list[0] != source or route_list[len(route_list)-1] != target:
            #print("Incorrect source or target")
            return False
        for i in range(len(route_list)):
            if i == len(route_list) - 1:
                break
            node1 = route_list[i]
            node2 = route_list[i+1]            
            if g.has_edge(node1,node2) == False:
                #print('non_existance connection')
                return False
        return True
    
    def find_random_reachable_nodes(self, graph):
        nodes = list(graph.nodes)
        while True:
            node_pair = random.sample(nodes, 2)
            if nx.has_path(graph, node_pair[0], node_pair[1]):
                return node_pair

    def generate_dataset(self, count=500):          
        G = pickle.load(open('source/DBPedia.pkl', 'rb'))
        all_walks = walker.random_walks(G, n_walks=1, walk_len = 1000, start_nodes=range(G.number_of_nodes()), alpha=0.2)
        
        for difficulty in ['easy', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (4, 19) if difficulty == 'easy' else (20, 50)        
            while len(self.problem_set) < count:
                node_size = sample_node_size(min_nodes, max_nodes)
                c = Counter(all_walks[random.randint(0, G.number_of_nodes()-1)])
                node_list = [k for k, v in c.most_common(node_size)]
                if len(node_list) < node_size:
                    continue
                H = nx.induced_subgraph(G, node_list).copy()
                if H.number_of_edges() < 3:
                    continue
                
                st, ds = self.find_random_reachable_nodes(H)
                exact_answer, path = self.exact_solver(H, st, ds)
                
                if len(self.examples) < 100:
                    self.examples.append(self.generate_example(H, path, st, ds))
                    continue
                
                self.problem_set.append({
                    'id' : len(self.problem_set),
                    'problem_text' : self.generate_problem(H, st, ds),
                    'graph': H,
                    'exact_answer': exact_answer,
                    'path': path,
                    'source': st,
                    'target': ds
                })
            self.save_dataset(difficulty)
        
    @staticmethod
    def exact_solver(graph, st, ds):
        shortest_path = nx.shortest_path(graph, source=st, target=ds)
        return len(shortest_path)-1, shortest_path
    
    def generate_problem(self, graph, st, ds):
        description = ["Your task is to identify the shortest path between two specified entities in an undirected knowledge graph, minimizing the number of hops."]
        description.append('\n**Problem to Solve**\n')
        description.append("- Entities in this knowledge graph: " + ", ".join(node['name'] for node in graph.nodes.values()))
        description.append("- The relationships between these entities are as follows:")
        for u, v, data in graph.edges(data=True):
            description.append(f" - {graph.nodes[u]['name']} is connected to {graph.nodes[v]['name']} via the relationship {data['relation']}.")
        
        description.append(f"Please determine the shortest path between {graph.nodes[st]['name']} and {graph.nodes[ds]['name']} in this network.")
        description.append("Submit your answer in the format: [Entity1, Entity2, ..., EntityN], where Entity1 and EntityN are the specified start and end entities, and Entity2 through EntityN-1 are the intermediate entities on the shortest path.")
        return '\n'.join(description)
    
    def generate_example(self, graph, path, st, ds):
        example = []
        example.append("- Entities in this knowledge graph: " + ", ".join(node['name'] for node in graph.nodes.values()))
        example.append("- The relationships between these entities are as follows:")
        for u, v, data in graph.edges(data=True):
            example.append(f" - {graph.nodes[u]['name']} is connected to {graph.nodes[v]['name']} via the relationship {data['relation']}.")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"One shortest path between {graph.nodes[st]['name']} and {graph.nodes[ds]['name']} is: [{answer}]")
        return '\n'.join(example)

