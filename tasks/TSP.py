import networkx as nx
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
import itertools
import pandas as pd
import fast_tsp
from tasks.base import *


class TSP_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(TSP_Task, self).__init__(data_loc, 'TSP')
        self.examples = []
        self.example_num = 0
        
    def check_solution(self, problem_id, response):
        g = self.problem_set[problem_id]['graph']      
        pattern = re.compile(r'\[\s*([A-Z\s,]*)\s*\]')
        p = pattern.findall(response)
        if p:
            matches = p[-1]
            matches = matches.split(",")
            route_list = [find_node_by_name(g, node.strip()) for node in matches]
            # check if all nodes are visitied exactly once
            # print(route_list)
            if set(route_list) == set(g.nodes()) and len(route_list) == len(g.nodes()) + 1 and route_list[0] == route_list[-1]:
                return self.compute_tour_length(g, route_list)
            return -2
        return -1
    
    @staticmethod
    def compute_tour_length(graph, route):
        tour_length = 0
        # assert route[0] == route[-1], "The route must start and end at the same node"
        for i in range(len(route) - 1):
            tour_length += graph.get_edge_data(route[i], route[i + 1])['weight']
        return tour_length

    def generate_dataset(self, count=500):
        G = nx.Graph()
        node_mapping = {}
        current_id = 0
        # load dataframe
        df = pd.read_csv('source/TSP.csv')
        for index, row in df.iterrows():
            source = row['source airport']
            destination = row['destination airport']
            weight = row['distance_km']
            
            # Assign an incremental identifier to each unique airport code
            if source not in node_mapping:
                node_mapping[source] = current_id
                G.add_node(current_id, name=source)
                current_id += 1

            if destination not in node_mapping:
                node_mapping[destination] = current_id
                G.add_node(current_id, name=destination)
                current_id += 1
            G.add_edge(node_mapping[source], node_mapping[destination], weight=weight)

        edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if math.isnan(data['weight'])]
        G.remove_edges_from(edges_to_remove)
        for u, v, data in G.edges(data=True):
            data['weight'] = int(data['weight'])

        # only use the largest connected component
        largest_component = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_component).copy()

        # compute the shortest path lengths between all node pairs
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2 and not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, weight=int(shortest_paths[node1][node2]))

        for difficulty in ['easy', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (4, 9) if difficulty == 'easy' else (10, 20)
            while len(self.problem_set) < count:            
                node_list = list(G.nodes())
                H = nx.induced_subgraph(G, random.sample(node_list, sample_node_size(min_nodes, max_nodes)))
                H = nx.relabel_nodes(H, {old_label: new_label for new_label, old_label in enumerate(H.nodes())})
                answer, path = self.exact_solver(H)
                if len(self.examples) < 100:
                    self.examples.append(self.generate_example(H, path))
                    continue
                problem_text = self.generate_problem(H)
                if len(problem_text) > 6000:
                    continue
                self.problem_set.append({
                    'id' : len(self.problem_set),
                    'problem_text': problem_text,
                    'graph': H,
                    'exact_answer': answer,
                    'path': path
                })
            self.save_dataset(difficulty)

    def generate_problem(self, graph):
        prompt = ["You are required to solve the Travelling Salesman Problem for an undirected flight route network. Your objective is to determine the shortest possible route that visits each of the listed airports exactly once and returns to the starting point."]
        prompt.append('\n**Problem to Solve**\n')
        prompt.append("- Airports to visit: " + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])),
        prompt.append("- Travel distances (in kilometers) between each pair of airports:")
        for edge in graph.edges(data=True):
            prompt.append(f"{graph.nodes[edge[0]]['name']} to {graph.nodes[edge[1]]['name']}: {edge[2]['weight']}")
        prompt.append("Please calculate the shortest tour and format your answer as follows: [Airport A, Airport B, Airport C, ..., Airport A]")
        return "\n".join(prompt)
    
    def generate_example(self, graph, path):
        example = []
        example.append('- Airports to visit: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        example.append("- Travel distances (in kilometers) between each pair of airports:")
        for edge in graph.edges(data=True):
            example.append(f"{graph.nodes[edge[0]]['name']} to {graph.nodes[edge[1]]['name']}: {edge[2]['weight']}")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"One shortest route: [{answer}].")
        return '\n'.join(example)

    def exact_solver(self, graph):
        dis_mat = self.build_distance_matrix(graph)
        route = fast_tsp.solve_tsp_exact(dis_mat)
        route.append(route[0])
        return self.compute_tour_length(graph, route), route
    
    def approx_solver(self, graph, method='greedy'):
        if method == 'random':
            route = list(np.random.permutation(graph.nodes()))
            route = route + [route[0]]
        elif method == 'greedy':
            route = nx.approximation.traveling_salesman_problem(graph, cycle=True, weight='weight', method=nx.approximation.greedy_tsp)
        elif method == 'approximated':
            route = nx.approximation.traveling_salesman_problem(graph, cycle=True, weight='weight', method=nx.approximation.christofides)
        return self.compute_tour_length(graph, route), route

    @staticmethod
    def build_distance_matrix(graph):
        n = len(graph)
        dist = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                dist[i][j] = graph.get_edge_data(i, j)['weight']
                dist[j][i] = graph.get_edge_data(i, j)['weight']
        return dist