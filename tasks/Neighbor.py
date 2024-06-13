import networkx as nx
from collections import Counter
import random
import re
import json
import numpy as np
import walker
import time
import itertools
import pandas as pd
from tasks.base import *
from tqdm import tqdm


class Neighbor_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(Neighbor_Task, self).__init__(data_loc, 'Neighbor')
        self.examples = []
    
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
                if self.is_feasible(g, node_list, problem_id):
                    return len(set(node_list))
                # else:
            return -2
        return -1

    def is_feasible(self, g, node_list, problem_id):
        node1 = self.problem_set[problem_id]['node1']
        node2 = self.problem_set[problem_id]['node2']  
        for node in node_list:
            if node == node1 or node == node2:
                return False
            if node not in g.neighbors(node1) or node not in g.neighbors(node2):
                return False
        return True
    
    def generate_dataset(self, count=500, difficulty='easy'): 
        with open('source/author.json', 'r') as f:
            node_names = json.load(f)
        G = nx.Graph()
        for node, name in node_names.items():
            G.add_node(int(node), name=name)
        with open('source/DBLP_2003_2018_5.txt', 'r') as f:
            for line in f:
                node1, node2 = map(int, line.strip().split())
                G.add_edge(node1, node2)
        print(f'graph statistics: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')         
        min_nodes, max_nodes = (4, 14) if difficulty == 'easy' else (15, 50)
        all_walks = walker.random_walks(G, n_walks=1, walk_len=1000, start_nodes=range(G.number_of_nodes()), alpha=0.2)
        
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
                
                for node1, node2 in itertools.combinations(H.nodes(), 2):
                    common_neighbors = set(H.neighbors(node1)).intersection(H.neighbors(node2))
                    if len(common_neighbors) >= 1:
                        nodes_with_common_neighbors = (node1, node2)
                        break
                    
                if nodes_with_common_neighbors is not None:
                    u,v = nodes_with_common_neighbors
                else:
                    continue
                
                exact_answer, path = self.exact_solver(H, u, v)
                
                if len(self.examples) < 100:
                    self.examples.append(self.generate_example(H, path, u, v))
                    continue
                
                self.problem_set.append({
                    'id' : len(self.problem_set),
                    'problem_text' : self.generate_problem(H, u, v),
                    'graph': H,
                    'path': path,
                    'exact_answer': exact_answer,
                    'node1': u,
                    'node2': v
                })
            self.save_dataset(difficulty)
            
    def generate_problem(self, graph, node1, node2):
        description = []
        description.append("Your task is to find the common neighbors of two nodes in an undirected academic network. In this network, nodes represent authors and edges represent research collaborations.")
        description.append('\n**Problem to Solve**\n')
        description.append("- Authors in the network: " + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()]))
        collaborations = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        description.append(f"- Research collaborations between these authors: {collaborations}.")
        description.append(f"Please identify the common neighbors of {graph.nodes[node1]['name']} and {graph.nodes[node2]['name']} in this network.")
        description.append('Present your answer in the following format: [AuthorA, AuthorB, AuthorC, AuthorD, ...].')
        return '\n'.join(description)

    def generate_example(self, graph, path, node1, node2):
        example = []
        example.append('- Authors in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()]))
        collaborations = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        example.append(f"- Research collaborations between these authors: {collaborations}.")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"Common neighbors between {graph.nodes[node1]['name']} and {graph.nodes[node2]['name']}: [{answer}]")
        return '\n'.join(example)
    
    @staticmethod
    def exact_solver(graph, u, v):
        common_neighbors = set(graph.neighbors(u)) & set(graph.neighbors(v))
        return len(common_neighbors), list(common_neighbors)