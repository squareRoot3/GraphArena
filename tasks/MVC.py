
from collections import Counter
import random
import pickle
import re
import json
import walker
from tasks.base import *


class MVC_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(MVC_Task, self).__init__(data_loc, 'MVC')
    
    def check_solution(self, problem_id, response):
        g = self.problem_set[problem_id]['graph']
        pattern = re.compile(r'\[(.*?)\]')
        p = pattern.findall(response)
        if p:
            matches = p[-1]
            matches = matches.split(",")
            author_list = [author.strip() for author in matches]
            node_list= []
            # print(len(author_list))
            for author in author_list:
                node = find_node_by_name(g, author)
                if node is None:
                    # print('Node not found:', author)
                    # return -2
                    continue
                node_list.append(node)
            return self.is_feasible(g, node_list)
        return -1
    
    def is_feasible(self, g, node_list):  
        for u, v in g.edges():
            # print(u, v, node_list)
            if u not in node_list and v not in node_list:
                return -2
        return len(node_list)
    
    def generate_dataset(self, count=500):
        G = pickle.load(open('source/social_network_union.pkl', 'rb'))
        all_walks = walker.random_walks(G, n_walks=1, walk_len = 1000, start_nodes=range(G.number_of_nodes()), alpha=0.2)
        self.examples = []
        for difficulty in ['easy', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (4, 14) if difficulty == 'easy' else (15, 30)
                
            while len(self.problem_set) < count:
                node_size = sample_node_size(min_nodes, max_nodes)
                c = Counter(random.choice(all_walks))
                node_list = [k for k, v in c.most_common(node_size)]
                H = nx.induced_subgraph(G, node_list)
                if len(node_list) < node_size or not nx.is_connected(H):
                    continue
                exact_answer, path = self.exact_solver(H)
                if exact_answer < 3:
                    continue
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
                    'exact_answer': exact_answer,
                    'path': path
                })         
            self.save_dataset(difficulty)
    
    def generate_problem(self, graph):
        prompt = ['Your task is to solve the Minimum Vertex Cover problem in the given social network. In this network, each node represents a user, and each edge represents a friendship connection. You need to identify the smallest subset of users such that every friendship connection has at least one user from this subset.']
        prompt.append('\n**Problem to Solve**\n')
        prompt.append('- Users in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        collaborations = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        prompt.append(f"- Fiendship connections: {collaborations}.")
        prompt.append("Identify the Minimum Vertex Cover of this network and present your answer in the following format: [UserA, UserB, UserC, UserD, ...].")
        return '\n'.join(prompt)
    
    def generate_example(self, graph, path):
        example = []
        example.append('- Users in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        collaborations = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        example.append(f"- Fiendship connections: {collaborations}.")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"One Minimum Vertex Cover: [{answer}].")
        return '\n'.join(example)

    @staticmethod 
    def exact_solver(graph):
        complement_G = nx.complement(graph)
        max_clique = max(nx.find_cliques(complement_G), key=len)
        max_independent_set = max_clique
        min_vertex_cover = set(graph.nodes()) - set(max_independent_set)
        return len(min_vertex_cover), min_vertex_cover
    
    def min_weighted_vertex_cover(self, G, weight=None):
        cost = dict(G.nodes(data=weight, default=1))
        cover = set()
        for u, v in G.edges():
            if u in cover or v in cover:
                continue
            if cost[u] <= cost[v]:
                cover.add(u)
                cost[v] -= cost[u]
            else:
                cover.add(v)
                cost[u] -= cost[v]
        return cover
    
    def approx_solver(self, graph, method='greedy'):
        if method == 'random': 
            min_vertex_cover= list(graph.nodes)

            while True:
                # Check if current set is a vertex cover
                if all(u in min_vertex_cover or v in min_vertex_cover for u, v in graph.edges):
                    # Randomly try to remove a node
                    node = random.choice(list(min_vertex_cover))
                    min_vertex_cover.remove(node)
                else:
                    # If it's not a vertex cover anymore, stop
                    min_vertex_cover.append(node)
                    break
            
        elif method == 'greedy':
            min_vertex_cover = []
            edges = list(graph.edges)

            # Sort edges by the sum of degrees of their endpoints, in descending order
            edges.sort(key=lambda edge: graph.degree(edge[0]) + graph.degree(edge[1]), reverse=True)

            while edges:
                # Select an edge
                u, v = edges.pop(0)
                # Add both endpoints to the cover
                min_vertex_cover.append(u)
                min_vertex_cover.append(v)
                # Remove all edges covered by u or v
                edges = [edge for edge in edges if u not in edge and v not in edge]

        elif method == 'approximated':
            min_vertex_cover = list(self.min_weighted_vertex_cover(graph,weight=None))
        return len(min_vertex_cover),min_vertex_cover

