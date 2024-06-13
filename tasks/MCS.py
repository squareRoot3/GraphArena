from tasks.base import *
import random
import pickle
import re
import networkx as nx
from itertools import product
from networkx.algorithms.approximation.clique import max_clique

class MCS_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(MCS_Task, self).__init__(data_loc, 'MCS')
        
    def check_solution(self, problem_id, response):
        pattern = r"\[\s*([\d\s,]+)\s*\]"
        # Search for the pattern in the response
        matches = re.findall(pattern, response)
        if len(matches) >= 2:
            try:
                molecule_a_indices = tuple(map(int, re.split(r'[\s,]+', matches[-2].strip())))
                molecule_b_indices = tuple(map(int, re.split(r'[\s,]+', matches[-1].strip())))
                if molecule_a_indices and molecule_b_indices and len(molecule_a_indices) == len(molecule_b_indices):
                    g1, g2 = self.problem_set[problem_id]['graph']
                    subgraph1 = g1.subgraph(molecule_a_indices)
                    subgraph2 = g2.subgraph(molecule_b_indices)
                    # print(subgraph1, subgraph2)
                    if nx.is_isomorphic(subgraph1, subgraph2):
                        return len(molecule_a_indices)
                return -2
            except Exception as e:
                print(e)
                return -1
        return -1

    @staticmethod
    def molecular_to_text(G, idx='A'):
        description = []
        nodes = G.nodes(data=True)
        edges = G.edges(data=True)
        description.append(f'Molecule {idx} consists of {len(nodes)} atoms with the following {len(edges)} bonds:')
        edges_discription = ', '.join([f'{u}-{v}' for u, v, data in edges]) +'.'
        description.append(edges_discription)
        return '\n'.join(description)

    def generate_dataset(self, count=100):
        molecular_set = pickle.load(open('source/molecular_graphs.pkl', 'rb'))
        self.examples = []
        for difficulty in ['easy', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (4, 9) if difficulty == 'easy' else (10, 20)
            while len(self.problem_set) < count:
                node_size1 = sample_node_size(min_nodes, max_nodes)
                node_size2 = sample_node_size(min_nodes, max_nodes)
                g1 = random.choice(molecular_set[node_size1])
                g2 = random.choice(molecular_set[node_size2])
                answer, path = None, None
                if difficulty == 'easy':
                    answer, path = self.exact_solver(g1, g2)
                if len(self.examples) < 100:
                    self.examples.append(self.generate_example(g1, g2, path))
                    continue
                self.problem_set.append({
                    'id': len(self.problem_set),
                    'problem_text': self.generate_problem(g1, g2),
                    'graph': (g1, g2),
                    'exact_answer': answer,
                    'path': path
                })
            self.save_dataset(difficulty)

    def generate_problem(self, g1, g2):
        prompt = ['You are required to solve the Maximum Common Subgraph problem. Your goal is to identify the common subgraph with the maximum number of atoms shared between the two molecules.']
        prompt.append('\n**Problem to Solve**\n')
        prompt.append('You are given the following two molecules:')
        prompt.append(self.molecular_to_text(g1, idx='A'))
        prompt.append(self.molecular_to_text(g2, idx='B'))
        prompt.append('Provide the indices of the atoms in the common subgraph for each molecule in the following format: [Node indices in molecular A], [Node indices in molecular B].')
        prompt.append('For example, if the common subgraph is the subgraph of atom 1, 2, 3 in molecule A and the subgrah of atom 2, 3, 4 in molecule B, you should answer: [1, 2, 3], [2, 3, 4].') 
        return '\n'.join(prompt)
        
    def generate_example(self, g1, g2, path):
        example = []
        example.append(self.molecular_to_text(g1, idx='A'))
        example.append(self.molecular_to_text(g2, idx='B'))
        indices_A = ', '.join([str(i) for i in path[0]])
        indices_B = ', '.join([str(i) for i in path[1]])
        example.append(f'One max common subgraph: [{indices_A}], [{indices_B}].') 
        return '\n'.join(example)

    @staticmethod
    def exact_solver(g1, g2):
        ismags = nx.isomorphism.ISMAGS(g1, g2)
        lcs = list(ismags.largest_common_subgraph(symmetry=True))[0]
        return len(lcs), (list(lcs.keys()), list(lcs.values()))

    def approx_solver(self, g1, g2):
        subgraph_sizes = []
        knt = 0
        while knt < 10:
            # Randomly pick a connected subgraph of g2
            nodes = random.sample(g2.nodes, k=10)
            subgraph = g2.subgraph(nodes)
            if nx.is_connected(subgraph):
                knt += 1
                subgraph_size, _ = self.exact_solver(g1, subgraph)
                subgraph_sizes.append(subgraph_size)

        return  max(subgraph_sizes)