from collections import Counter
import random
import pickle
import re
import json
import walker
from tasks.base import *


class MIS_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(MIS_Task, self).__init__(data_loc, 'MIS')
    
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
            g_sub = nx.induced_subgraph(g, node_list)
            if g_sub.number_of_edges()==0:
                return len(node_list)
            return -2
        return -1
    
    def generate_dataset(self, count=500):          
        G = pickle.load(open('source/social_network.pkl', 'rb'))
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
        prompt = ['Your task is to solve the Maximum Independent Set problem in the given social network. In this network, each node represents a user, and each edge represents a friendship connection. You need to identify the largest subset of users such that no two users in this subset are friends connected by an edge. ']
        prompt.append('\n**Problem to Solve**\n')
        prompt.append('- Users in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        collaborations = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        prompt.append(f"- Fiendship connections: {collaborations}.")
        prompt.append("Identify the Maximum Independent Set of this network and present your answer in the following format: [UserA, UserB, UserC, UserD, ...].")
        return '\n'.join(prompt)

    def generate_example(self, graph, path):
        example = []
        example.append('- Users in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        collaborations = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        example.append(f"- Fiendship connections: {collaborations}.")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"One Maximum Independent Set: [{answer}].")
        return '\n'.join(example)
    
    def exact_solver(self, graph):
                    #complement graph of H
        complement_G = nx.complement(graph)
        max_clique = max(nx.find_cliques(complement_G), key=len)
        return len(max_clique), max_clique
 
    def approx_solver(self, graph, method='greedy'):
        if method == 'random': 
            nodes = list(graph.nodes)
            random.shuffle(nodes)
            independent_set = []

            for node in nodes:
                # Check if the node can be added to the independent set
                if all(neighbor not in independent_set for neighbor in graph.neighbors(node)):
                    independent_set.append(node)
                else:
                    break

        elif method == 'greedy':
            nodes = sorted(graph.nodes, key=lambda x: graph.degree(x))
            independent_set = []
            for node in nodes:
                # Check if the node can be added to the independent set
                if not any(neighbor in independent_set for neighbor in graph.neighbors(node)):
                    independent_set.append(node)

        elif method == 'chris':
            independent_set = list(nx.approximation.maximum_independent_set(graph))
            
        return len(independent_set),independent_set

if __name__ == '__main__':
    task = MIS_Task('NPdataset')
    task.generate_dataset()
    
