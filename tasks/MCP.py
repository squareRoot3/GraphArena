from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
import networkx as nx
from collections import Counter
import random
import re
import json
import numpy as np
import walker
import pandas as pd
from tasks.base import *


class MCP_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(MCP_Task, self).__init__(data_loc, 'MCP')

    def check_solution(self, problem_id, response):
        g = self.problem_set[problem_id]['graph']
        pattern = re.compile(r'\[(.*?)\]')
        p = pattern.findall(response)
        if p:
            matches = p[-1]
            matches = matches.split(",")
            author_list = [author.strip() for author in matches]
            node_list= []
            for author in author_list:
                node = find_node_by_name(g, author)
                if node is None:
                    # print('Node not found:', author)
                    # return -2
                    continue
                node_list.append(node)
            g_sub = nx.induced_subgraph(g, node_list)
            # print('MVC', g_sub)
            if g_sub.number_of_edges() >= len(node_list) * (len(node_list) - 1) / 2:
                return len(node_list)
            return -2
        return -1
    
    def generate_dataset(self, count=100, difficulty='easy'):
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
        all_walks = walker.random_walks(G, n_walks=1, walk_len = 1000, start_nodes=range(G.number_of_nodes()), alpha=0.2)

        self.examples = []
        for difficulty in ['easy', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (4, 14) if difficulty == 'easy' else (15, 30)

            while len(self.problem_set) < count:
                node_size = sample_node_size(min_nodes, max_nodes)
                # randomly select a walk
                c = Counter(random.choice(all_walks))
                node_list = [k for k, v in c.most_common(node_size)]
                if len(node_list) < node_size:
                    continue
                H = nx.induced_subgraph(G, node_list)
                answer, path = self.exact_solver(H)
                if len(self.examples) < 100:
                    self.examples.append(self.generate_example(H, path))
                    continue                
                problem_text = self.generate_problem(H)
                if len(problem_text) > 6000: 
                    continue
                self.problem_set.append({
                    'id':len(self.problem_set),
                    'problem_text': problem_text,
                    'graph': H,
                    'exact_answer': answer,
                    'path': path
                })    
            self.save_dataset(difficulty)

    def generate_problem(self, graph):
        prompt = ['You are required to solve the Maximum Clique Problem for an undirected academic network. In this network, nodes represent authors and edges represent research collaborations. Your objective is to find the largest subset of nodes such that every pair of vertices in this subset is connected by an edge.']
        prompt.append('\n**Problem to Solve**\n')
        prompt.append('- Authors in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()]))
        collaborations = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        prompt.append(f"- Research collaborations between these authors: {collaborations}.")
        prompt.append("Identify the clique with the maximum number of authors in this network. Present your answer in the following format: [AuthorA, AuthorB, AuthorC, AuthorD, ...].")
        return '\n'.join(prompt)

    def generate_example(self, graph, path):
        example = []
        example.append('- Authors in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        collaborations = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        example.append(f"- Research collaborations between these authors: {collaborations}.")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"One Maximum Clique: [{answer}].")
        return '\n'.join(example)

    @staticmethod
    def exact_solver(graph):
        clique = max(nx.find_cliques(graph), key=len)
        return len(clique), clique
