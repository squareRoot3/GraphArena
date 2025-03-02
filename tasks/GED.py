from collections import Counter
from tasks.base import *
import random
import pickle
import re
from munkres import Munkres


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
                    # print('gmap', gmap)  # Debug: print the gmap
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
    
    def generate_example(self, g1, g2):
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

    def compute_edit_cost(self, g1, g2):
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

    def approx_solver(self, g, timeout=0.1, method='greedy'):
        g1, g2 = g
        if method == 'random':
            node_list = list(g1.nodes())
            g1 = nx.relabel_nodes(g1, dict(zip(node_list, random.sample(node_list, len(node_list)))))
            return self.compute_edit_cost(g1, g2)
        if method == 'greedy':
            return VJ(g1, g2)[1]
        if method == 'approximated':
            return graph_edit_distance(g1, g2)[1]


def VJ(g1,g2):
    edit_path = []
    g1_nodes = []
    g2_nodes = []
    for node in g1.nodes():
        g1_nodes.append((node,g1.nodes[node]['label']))
    for node in g2.nodes():
        g2_nodes.append((node,g2.nodes[node]['label']))
    g1_nodes = sorted(g1_nodes, key=lambda x: (x[1],x[0]))
    g2_nodes = sorted(g2_nodes, key=lambda x: (x[1],x[0]))
    g1_nodes_tmp = g1_nodes.copy()
    g2_nodes_tmp = g2_nodes.copy()
    i = 0
    j = 0
    while (i<len(g1_nodes) and j<len(g2_nodes)):
        if g1_nodes[i][1] == g2_nodes[j][1]:
            edit_path.append((g1_nodes[i][0],g2_nodes[j][0]))
            i += 1
            j += 1
            del g1_nodes_tmp[i-len(g1_nodes)-1]
            del g2_nodes_tmp[j-len(g2_nodes)-1]
        elif g1_nodes[i][1] > g2_nodes[j][1]:
            j += 1
        else:
            i += 1

    if (len(g1_nodes_tmp) == len(g2_nodes_tmp)):
        for k in range(len(g1_nodes_tmp)):
            edit_path.append((g1_nodes_tmp[k][0], g2_nodes_tmp[k][0]))
    if (len(g1_nodes_tmp) > len(g2_nodes_tmp)):
        for k in range(len(g2_nodes_tmp)):
            edit_path.append((g1_nodes_tmp[k][0], g2_nodes_tmp[k][0]))
        for k in range(len(g2_nodes_tmp),len(g1_nodes_tmp)):
            edit_path.append((g1_nodes_tmp[k][0], None))
    if (len(g1_nodes_tmp) < len(g2_nodes_tmp)):
        for k in range(len(g1_nodes_tmp)):
            edit_path.append((g1_nodes_tmp[k][0], g2_nodes_tmp[k][0]))
        for k in range(len(g1_nodes_tmp),len(g2_nodes_tmp)):
            edit_path.append((None, g2_nodes_tmp[k][0]))
    edit_path_tmp = []
    cost_list = []
    for i in edit_path:
        edit_path_tmp.append(i)
        cost_list.append(cost_edit_path(edit_path_tmp, g1, g2, 'VJ'))
    return edit_path, cost_edit_path(edit_path, g1, g2, 'VJ'), cost_list

class DFS_hungary():
    def __init__(self, g1, g2):
        self.g1, self.g2=g1, g2
        self.nx = list(self.g1.nodes())
        self.ny = list(self.g2.nodes())
        self.edge = {}
        for node1 in self.g1:
            edge_tmp = {}
            for node2 in self.g2:
                if self.g1.nodes[node1]['label'] == self.g2.nodes[node2]['label']:
                    edge_tmp[node2] = 0
                else:
                    edge_tmp[node2] = 1
            self.edge[node1] = edge_tmp
        self.cx = {}
        for node in self.g1:
            self.cx[node] = -1
        self.cy = {}
        self.visited = {}
        for node in self.g2:
            self.cy[node] = -1
            self.visited[node] = 0
        self.edit_path = []

    def min_cost(self):
        res=0
        for i in self.nx:
            if self.cx[i]==-1:
                for key in self.ny:
                    self.visited[key]=0
                res+=self.path(i)
        return res,self.edit_path

    def path(self, u):
        for v in self.ny:
            if not(self.edge[u][v]) and (not self.visited[v]):
                self.visited[v]=1
                if self.cy[v]==-1:
                    self.cx[u] = v
                    self.cy[v] = u
                    self.edit_path.append((u,v))
                    return 0
                else:
                    self.edit_path.remove((self.cy[v], v))
                    if not (self.path(self.cy[v])):
                        self.cx[u] = v
                        self.cy[v] = u
                        self.edit_path.append((u, v))
                        return 0
        self.edit_path.append((u,None))
        return 1

def Hungarian(g1,g2):
    cost, edit_path = DFS_hungary(g1,g2).min_cost()
    if len(g1.nodes()) < len(g2.nodes()):
        processed = [v[1] for v in edit_path]
        for node in g2.nodes():
            if node not in processed:
                edit_path.append((None,node))
    edit_path_tmp = []
    cost_list = []
    for i in edit_path:
        edit_path_tmp.append(i)
        cost_list.append(cost_edit_path(edit_path_tmp, g1, g2, 'Hungarian'))
    return edit_path, cost_edit_path(edit_path, g1, g2, 'Hungarian'), cost_list

def cost_edit_path(edit_path, u, v, lower_bound):
    cost = 0
    source_nodes = []
    target_nodes = []
    nodes_dict = {}
    for operation in edit_path:
        if operation[0] == None:
            cost += 1
            target_nodes.append(operation[1])
        elif operation[1] == None:
            cost += 1
            source_nodes.append(operation[0])
        else:
            if u.nodes[operation[0]]['label'] != v.nodes[operation[1]]['label']:
                cost += 1
            source_nodes.append(operation[0])
            target_nodes.append(operation[1])
        nodes_dict[operation[0]] = operation[1]

    edge_source = u.subgraph(source_nodes).edges()
    edge_target = v.subgraph(target_nodes).edges()

    sum = 0
    for edge in list(edge_source):
        (p, q) = (nodes_dict[edge[0]], nodes_dict[edge[1]])
        if (p, q) in edge_target:
            sum += 1
    cost = cost + len(edge_source) + len(edge_target) - 2 * sum

    if len(lower_bound) == 3 and lower_bound[2] == 'a':
        # Anchor
        anchor_cost = 0
        cross_edge_source = []
        cross_edge_target = []
        cross_edge_source_tmp = set(u.edges(source_nodes))
        for edge in cross_edge_source_tmp:
            if edge[0] not in source_nodes or edge[1] not in source_nodes:
                cross_edge_source.append(edge)
        cross_edge_target_tmp = set(v.edges(target_nodes))
        for edge in cross_edge_target_tmp:
            if edge[0] not in target_nodes or edge[1] not in target_nodes:
                cross_edge_target.append(edge)

        for edge in cross_edge_source:
            (p, q) = (nodes_dict[edge[0]], edge[1])
            if (p, q) in cross_edge_target:
                anchor_cost += 1

        return cost + anchor_cost
    else:
        return cost
    
def graph_edit_distance(u, v, beam_size = 10, lower_bound = 'SM', start_node = None):
    # Partial edit path
    open_set = []
    cost_open_set = []
    partial_cost_set = []
    path_idx_list = []
    time_count = 0.0
    # For each node w in V2, insert the substitution {u1 -> w} into OPEN
    if start_node == None or start_node not in list(u.nodes()):
        u1 = list(u.nodes())[0] # randomly access a node
    else:
        u1 = start_node
    call_count = 0
    unprocessed_u_set = []
    unprocessed_v_set = []
    for w in list(v.nodes()):
        edit_path = []
        edit_path.append((u1, w))
        unprocessed_u, unprocessed_v = check_unprocessed(u, v, edit_path)
        new_cost = cost_edit_path(edit_path, u, v, lower_bound)
        cost_list = [new_cost]
        unprocessed_u_set.append(unprocessed_u)
        unprocessed_v_set.append(unprocessed_v)
        # new_cost += unprocessed_cost(unprocessed_u, unprocessed_v, u, v)
        call_count += 1
        open_set.append(edit_path)
        partial_cost_set.append(cost_list)
    unprocessed_cost_set = unprocessed_cost(unprocessed_u_set, unprocessed_v_set, lower_bound, u, v)
    start = time.time()
    for i in range(len(unprocessed_cost_set)):
        new_cost = unprocessed_cost_set[i] + partial_cost_set[i][0]
        cost_open_set.append(new_cost)
    end = time.time()
    time_count = time_count + end - start

    # Insert the deletion {u1 -> none} into OPEN
    edit_path = []
    edit_path.append((u1, None))
    unprocessed_u, unprocessed_v = check_unprocessed(u, v, edit_path)
    new_cost = cost_edit_path(edit_path, u, v, lower_bound)
    cost_list = [new_cost]
    start = time.time()
    new_cost_set = unprocessed_cost([unprocessed_u], [unprocessed_v], lower_bound, u, v)
    new_cost += new_cost_set[0]
    end = time.time()
    time_count = time_count + end - start
    call_count += 1
    open_set.append(edit_path)
    cost_open_set.append(new_cost)
    partial_cost_set.append(cost_list)
    
    while cost_open_set:
        if beam_size:
            # BeamSearch
            tmp_path_set = []
            tmp_cost_set = []
            tmp_partial_cost_set = []
            if len(cost_open_set) > beam_size:
                zipped = zip(open_set,cost_open_set,partial_cost_set)
                sort_zipped = sorted(zipped, key = lambda x: x[1])
                result = zip(*sort_zipped)
                open_set, cost_open_set, partial_cost_set = [list(x)[0:beam_size] for x in result]
        path_idx = cost_open_set.index(min(cost_open_set))
        path_idx_list.append(path_idx)
        min_path = open_set.pop(path_idx)
        cost = cost_open_set.pop(path_idx)
        cost_list = partial_cost_set.pop(path_idx)

        # print(len(open_set))
        # Check p_min is a complete edit path
        unprocessed_u, unprocessed_v = check_unprocessed(u, v, min_path)

        # Return if p_min is a complete edit path
        if not unprocessed_u and not unprocessed_v:
            return min_path, cost, cost_list, call_count, time_count, path_idx_list

        else:
            if unprocessed_u:
                u_next = unprocessed_u.pop()
                unprocessed_u_set = []
                unprocessed_v_set = []
                for v_next in unprocessed_v:
                    new_path = min_path.copy()
                    new_path.append((u_next, v_next))
                    unprocessed_u, unprocessed_v = check_unprocessed(u, v, new_path)
                    new_cost = cost_edit_path(new_path, u, v, lower_bound)
                    new_cost_list = cost_list.copy()
                    new_cost_list.append(new_cost)
                    unprocessed_u_set.append(unprocessed_u)
                    unprocessed_v_set.append(unprocessed_v)
                    # new_cost += unprocessed_cost(unprocessed_u, unprocessed_v, u, v)
                    call_count += 1
                    open_set.append(new_path)
                    # cost_open_set.append(new_cost)
                    partial_cost_set.append(new_cost_list)
                start = time.time()
                new_cost_set = unprocessed_cost(unprocessed_u_set, unprocessed_v_set, lower_bound, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i-len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.time()
                time_count = time_count + end - start
                
                new_path = new_path = min_path.copy()
                new_path.append((u_next, None))
                unprocessed_u, unprocessed_v = check_unprocessed(u, v, new_path)
                new_cost = cost_edit_path(new_path, u, v, lower_bound)
                new_cost_list = cost_list.copy()
                new_cost_list.append(new_cost)
                start = time.time()
                new_cost_set = unprocessed_cost([unprocessed_u], [unprocessed_v], lower_bound, u, v)
                new_cost += new_cost_set[0]
                end = time.time()
                time_count = time_count + end - start
                call_count += 1
                open_set.append(new_path)
                cost_open_set.append(new_cost)
                partial_cost_set.append(new_cost_list)


            else:
                # All nodes in u have been processed, all nodes in v should be Added.
                unprocessed_u_set = []
                unprocessed_v_set = []
                for v_next in unprocessed_v:
                    new_path = min_path.copy()
                    new_path.append((None, v_next))
                    new_cost = cost_edit_path(new_path, u, v, lower_bound)
                    new_cost_list = cost_list.copy()
                    new_cost_list.append(new_cost)
                    unprocessed_u_set.append(unprocessed_u)
                    unprocessed_v_set.append(unprocessed_v)
                    call_count += 1
                    open_set.append(new_path)
                    # cost_open_set.append(new_cost)
                    partial_cost_set.append(new_cost_list)
                start = time.time()
                new_cost_set = unprocessed_cost(unprocessed_u_set, unprocessed_v_set, lower_bound, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i-len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.time()
                time_count = time_count + end - start
    return None, None, None, None, None, None

def check_unprocessed(u, v, path):
    processed_u = []
    processed_v = []

    for operation in path:
        if operation[0] != None:
            processed_u.append(operation[0])

        if operation[1] != None:
            processed_v.append(operation[1])
    # print(processed_u, processed_v)
    unprocessed_u = set(u.nodes()) - set(processed_u)
    unprocessed_v = set(v.nodes()) - set(processed_v)
    return list(unprocessed_u), list(unprocessed_v)


def list_unprocessed_label(unprocessed_node, u):
    unprocessed_label = []
    for node in unprocessed_node:
        unprocessed_label.append(u.nodes[node]['label'])
    unprocessed_label.sort()
    return unprocessed_label

def star_cost(p,q):
    cost = 0
    if p == None:
        cost += 2 * len(q) - 1
        return cost
    if q == None:
        cost += 2 * len(p) - 1
        return cost
    if p[0] != q[0]:
        cost += 1
    if len(p) > 1 and len(q) > 1:
        p[1:].sort()
        q[1:].sort()
        i = 1
        j = 1
        cross_node = 0
        while (i < len(p) and j < len(q)):
            if p[i] == q[j]:
                cross_node += 1
                i += 1
                j += 1
            elif p[i] < q[j]:
                i += 1
            else:
                j += 1
        cost = cost + max(len(p),len(q)) - 1 - cross_node
    cost += abs(len(q)-len(p))
    return cost


def unprocessed_cost(unprocessed_u_set, unprocessed_v_set, lower_bound, u, v):
    if lower_bound == 'heuristic':
        # heuristic
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            inter_node = set(unprocessed_u).intersection(set(unprocessed_v))
            cost = max(len(unprocessed_u), len(unprocessed_v)) - len(inter_node)
            cost_set.append(cost)
        return cost_set
    elif lower_bound[0:2] == 'LS':
        # LS
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            cross_node = 0
            u_label = list_unprocessed_label(unprocessed_u, u)
            v_label = list_unprocessed_label(unprocessed_v, v)

            i = 0
            j = 0
            while (i < len(u_label) and j < len(v_label)):
                if u_label[i] == v_label[j]:
                    cross_node += 1
                    i += 1
                    j += 1
                elif u_label[i] < v_label[j]:
                    i += 1
                else:
                    j += 1
            node_cost = max(len(unprocessed_u), len(unprocessed_v)) - cross_node
            edge_u = u.subgraph(unprocessed_u).edges()
            edge_v = v.subgraph(unprocessed_v).edges()
            inter_edge = set(edge_u).intersection(set(edge_v))
            edge_cost = max(len(edge_u), len(edge_v)) - len(inter_edge)
            cost = node_cost + edge_cost
            cost_set.append(cost)
        return cost_set

    elif lower_bound == 'BM':
    # BM
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            cost = 0
            u_label = list_unprocessed_label(unprocessed_u, u)
            v_label = list_unprocessed_label(unprocessed_v, v)
            i = 0
            j = 0
            while (i < len(u_label) and j < len(v_label)):
                if u_label[i] == v_label[j] and u.edges(unprocessed_u[i]) == v.edges(unprocessed_v[j]):
                    u_label.pop(i)
                    v_label.pop(j)
                    i += 1
                    j += 1
                elif u_label[i] < v_label[j]:
                    i += 1
                else:
                    j += 1
            i = 0
            j = 0
            while (i < len(u_label) and j < len(v_label)):
                if u_label[i] == v_label[j]:
                    cost += 0.5
                    u_label.pop(i)
                    v_label.pop(j)
                    i += 1
                    j += 1
                elif u_label[i] < v_label[j]:
                    i += 1
                else:
                    j += 1
            cost = cost + max(len(u_label), len(v_label))
            cost_set.append(cost)
        return cost_set
    else:
        #SM
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            stars_u = []
            temp_u = u.subgraph(unprocessed_u)
            for node in unprocessed_u:
                node_list = []
                node_list.append(node)
                for k in temp_u.neighbors(node):
                    node_list.append(k)
                stars_u.append(node_list)
            
            stars_v = []
            temp_v = v.subgraph(unprocessed_v)
            for node in unprocessed_v:
                node_list = []
                node_list.append(node)
                for k in temp_v.neighbors(node):
                    node_list.append(k)
                stars_v.append(node_list)

            max_degree = 0
            for i in stars_u:
                if len(i) > max_degree:
                    max_degree = len(i)

            for i in stars_v:
                if len(i) > max_degree:
                    max_degree = len(i)
            # Initial cost matrix
            if len(stars_u) > len(stars_v):
                for i in range(len(stars_u)-len(stars_v)):
                    stars_v.append(None)
            if len(stars_u) < len(stars_v):
                for i in range(len(stars_v)-len(stars_u)):
                    stars_u.append(None)
            cost_matrix = []
            for star1 in stars_u:
                cost_tmp = []
                for star2 in stars_v:
                    cost_tmp.append(star_cost(star1,star2))
                cost_matrix.append(cost_tmp)
            if cost_matrix == []:
                cost_set.append(0)
            else:
                m = Munkres()
                indexes = m.compute(cost_matrix)
                cost = 0
                for row, column in indexes:
                    value = cost_matrix[row][column]
                    cost += value
                cost = cost / max(4,max_degree)
                cost_set.append(cost)
        return cost_set