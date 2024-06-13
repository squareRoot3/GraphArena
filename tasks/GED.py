from collections import Counter
from tasks.base import *
import random
import pickle
import re


class GED_Task(NPTask):
    def __init__(self, data_loc='dataset', task_name='GED', example_num=0):
        super(GED_Task, self).__init__(data_loc, task_name)
        self.examples = []  # todo: add example
        self.example_num = example_num

    def check_solution(self, problem_id, response):
        g1, g2 = self.problem_set[problem_id]['graph']
        pattern = r"\[\s*([\d\s,]+)\s*\]"
        matches = re.findall(pattern, response)
        
        if matches:
            try:
                for match in reversed(matches):
                # match = matches[-1]
                    gmap = tuple(map(int, re.split(r'[\s,]+', match.strip())))
                    print('gmap', gmap)  # Debug: print the gmap
                    if set(gmap) == set(range(g1.number_of_nodes())) and len(gmap) == g1.number_of_nodes():
                        g1 = nx.relabel_nodes(g1, dict(zip(range(g1.number_of_nodes()), gmap)))
                        edit_cost = 0
                        for i in range(g1.number_of_nodes()):
                            edit_cost += g1.nodes[i]['label'] != g2.nodes[i]['label']                   
                        for edge in g1.edges():
                            if edge not in g2.edges():
                                edit_cost += 1
                        for edge in g2.edges():
                            if edge not in g1.edges():        
                                edit_cost += 1
                        return edit_cost
            except Exception as e:
                print(e)
                return -2
            return -2
        return -1
    
    @staticmethod
    def molecular_to_text(G, idx='A'):
        description = []
        # Describe atoms
        nodes = G.nodes(data=True)
        atom_descriptions = [f'{data["label"]} (atom {node})' for node, data in nodes]
        description.append(f'Molecule {idx}:')
        description.append(f'- Atoms: {", ".join(atom_descriptions)}.')
        edges = G.edges(data=True)
        description.append(f'- Bonds: {", ".join([f"{u}-{v}" for u, v, data in edges])}.')
        return '\n'.join(description)

    def generate_dataset(self, count=100):
        molecular_set = pickle.load(open('source/molecular_graphs.pkl', 'rb'))
        for difficulty in ['easy', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (4, 9) if difficulty == 'easy' else (10, 20)
            while len(self.problem_set) < count:
                node_size = sample_node_size(min_nodes, max_nodes)
                g1, g2 = random.sample(molecular_set[node_size], 2)
                # print(len(self.problem_set))
                answer, path = None, None
                if difficulty == 'easy':
                    answer, path = self.exact_solver(g1, g2)
                if len(self.examples) < 100:
                    self.examples.append(self.generate_example(g1, g2, path))
                    continue
                self.problem_set.append({
                    'id': len(self.problem_set),
                    'problem_text': self.generate_problem(g1,g2),
                    'graph': (g1, g2),
                    'exact_answer': answer,
                    'path': path,
                })
            self.save_dataset(difficulty)
        
    def generate_problem(self, g1, g2):
        prompt = ['You are required to solve the Graph Edit Distance problem between two molecules. Each edit operation (adding or deleting an edge, adding or deleting an isolated node, or relabeling a node) has the identity cost. Your objective is to establish a mapping between the atom IDs from Molecule A to Molecule B, ensuring that each atom ID in Molecule A corresponds to exactly one atom ID in Molecule B. The mapping corresponds to the minimum edit cost between the two graphs.']
        prompt.append('\n**Problem to Solve**\n')
        prompt.append('You are given the following two molecules:')
        prompt.append(self.molecular_to_text(g1, idx='A'))
        prompt.append(self.molecular_to_text(g2, idx='B'))
        prompt.append('Represent the node mapping as a list of integers, where the position in the list corresponds to the atom ID in Molecule A and the value at that position indicates the corresponding atom ID in Molecule B.')
        prompt.append('For instance, if atom 0 in Molecule A corresponds to atom 1 in Molecule B, atom 1 in Molecule A corresponds to atom 0 in Molecule B, and atom 2 remains unchanged, the mapping would be represented as [1, 0, 2, ...].')
        return '\n'.join(prompt)
    
    def generate_example(self, g1, g2, path):
        example = []
        example.append(self.molecular_to_text(g1, idx='A'))
        example.append(self.molecular_to_text(g2, idx='B'))
        mapping = ', '.join([str(i) for i in path])
        example.append(f'One optimal node mapping: [{mapping}].') 
        return '\n'.join(example)

    def exact_solver(self, g1, g2):
        path, answer = nx.optimal_edit_paths(g1, g2, node_subst_cost = lambda x, y: 0 if x['label'] == y['label'] else 1)
        ind = [None,] * g1.number_of_nodes()
        for u, v in path[0][0]:
            ind[u] = v
        return answer, ind
    
    def approx_solver(self, g1, g2, timeout=10):
        return nx.graph_edit_distance(g1, g2, node_subst_cost = lambda x, y: 0 if x['label'] == y['label'] else 1, timeout=timeout)