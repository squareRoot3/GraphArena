from tasks import *
from openai import OpenAI
import networkx as nx
import random
import numpy as np
import os
import argparse
import dgl
import torch
from dgl.nn import GraphConv, GINConv, GATConv, SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pickle


class GPDataset(DGLDataset):
    def __init__(self, task='Connected', mode='train', difficulty='easy'):
        super().__init__(name='GP')
        dataset_loc = 'dataset_train' if mode == 'train' else 'dataset'
        task = globals()[task+'_Task'](dataset_loc)
        task.load_dataset(difficulty)
        graph_problems = task.problem_set
        if mode == 'train':
            task.load_dataset('hard')
            graph_problems += task.problem_set
        print(len(graph_problems))
        self.graphs = []
        self.labels = []
        for gp in graph_problems:
            if gp['exact_answer'] == None:
                continue
            if len(gp['graph']) == 2:
                gp['graph'] = nx.disjoint_union(gp['graph'][0], gp['graph'][1])
            node_dict = {j:i for i, j in enumerate(gp['graph'].nodes())}
            g = dgl.from_networkx(gp['graph'])
            g.ndata['feat'] = torch.ones(g.number_of_nodes(),1)
            self.labels.append(gp['exact_answer'])
            if 'node1' in gp:
                gp['source'] = gp['node1']
            if 'node2' in gp:
                gp['target'] = gp['node2']
            if 'source' in gp:
                nx.relabel_nodes(gp['graph'], {i: str(i) for i in gp['graph'].nodes()}, copy=False)
                g = dgl.from_networkx(gp['graph'])
                g.ndata['feat'] = torch.ones(g.number_of_nodes(),1)
                gp['source'] = node_dict[gp['source']]
                g.ndata['source'] = torch.zeros(g.number_of_nodes(), 1).bool()
                g.ndata['source'][gp['source']] = True
            if 'target' in gp:
                gp['target'] = node_dict[gp['target']]
                g.ndata['target'] = torch.zeros(g.number_of_nodes(), 1).bool()
                g.ndata['target'][gp['target']] = True
            g = dgl.add_self_loop(g)
            self.graphs.append(g)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class GNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, st_nodes=False, conv_type='GIN', act=nn.LeakyReLU()):
        super(GNN, self).__init__()
        if conv_type == 'GCN':
            self.conv1 = GraphConv(in_feats, h_feats, activation=act)
            self.conv2 = GraphConv(h_feats, h_feats, activation=act)
        elif conv_type == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats, num_heads=1, activation=act)
            self.conv2 = GATConv(h_feats, h_feats, num_heads=1, activation=act)
        elif conv_type == 'GIN':
            self.conv1 = GINConv(nn.Linear(in_feats, h_feats), activation=act, aggregator_type='sum')
            self.conv2 = GINConv(nn.Linear(h_feats, h_feats), activation=act, aggregator_type='sum')
        elif conv_type == 'SAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, 'pool', activation=act)
            self.conv2 = SAGEConv(h_feats, h_feats, 'pool', activation=act)
        self.st_nodes = st_nodes
        self.conv_type = conv_type
        if st_nodes:
            h_feats = h_feats*3
        self.linear = nn.Linear(h_feats, num_classes)

    def forward(self, g, in_feat, e_feat=None):
        h = self.conv1(g, in_feat, edge_weight=e_feat)
        h = self.conv2(g, h.reshape(h.shape[0], -1), edge_weight=e_feat)
        h = h.reshape(h.shape[0], -1)
        g.ndata["h"] = h
        h = dgl.sum_nodes(g, "h")
        if self.st_nodes:
            h_source = g.ndata["h"][g.ndata['source'].squeeze(-1)]
            h_target = g.ndata["h"][g.ndata['target'].squeeze(-1)]
            h = torch.concat([h, h_source, h_target], dim=1)
        return self.linear(h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GNN experiments.")
    parser.add_argument("--model",
                        type=str,
                        default="GIN",
                        help='model name. candidates: [GIN, GAT, SAGE]')
    parser.add_argument("--task",
                    type=str,
                    default="Distance",
                    help='task name. candidates: [' + ', '.join(['Distance', 'Neighbor', 'Diameter', 'Connected']) + ']')
    parser.add_argument("--device", type=str, default="cuda:7", help="device to run the model")
    parser.add_argument("--h_feat", type=int, default=16, help="hidden feature size")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    args = parser.parse_args()
    
    task = args.task
    device = args.device
    h_feat = args.h_feat
    st_nodes = True if task in ['Neighbor', 'Distance'] else False
    out_feat = 1
    train_set = GPDataset(task, 'train')

    test_set_easy = GPDataset(task, 'test', 'easy')
    test_set_hard = GPDataset(task, 'test', 'hard')

    train_sampler = SubsetRandomSampler(torch.arange(len(train_set)))
    test_sampler = SubsetRandomSampler(torch.arange(len(test_set_easy)))
    train_dataloader = GraphDataLoader(
        train_set, sampler=train_sampler, batch_size=32, drop_last=False
    )
    model = GNN(1, h_feat, out_feat, st_nodes=st_nodes, conv_type=args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(args.epochs):
        loss_list = []
        for batched_graph, labels in train_dataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            e_feat = None
            if 'feat' in batched_graph.edata:
                e_feat = batched_graph.edata["feat"].float()
            pred = model(batched_graph, batched_graph.ndata["feat"].float(), e_feat)
            if out_feat == 1:
                loss = F.mse_loss(pred.squeeze(-1), labels.float())
            else:
                loss = F.cross_entropy(pred, labels)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: {}, loss: {:.3f}'.format(epoch, np.mean(loss_list)))
    
    
    
    num_correct, num_tests = 0, 0
    test_dataloader = GraphDataLoader(
        test_set_easy, sampler=test_sampler, batch_size=32, drop_last=False
    )
    for batched_graph, labels in test_dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        e_feat = None
        if 'feat' in batched_graph.edata:
            e_feat = batched_graph.edata["feat"].float()
        pred = model(batched_graph, batched_graph.ndata["feat"].float(), e_feat)
        if out_feat == 1:
            pred = torch.round(pred)
            num_correct += (pred.squeeze(-1) == labels).sum().item()
        else:
            num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)
    with open('GNN_results.txt', '+a') as f: 
        f.write("Task: {}, Model: {}, Test easy accuracy: {:.3f}\n".format(task, args.model, num_correct / num_tests))
    print("Task: {}, Model: {}, Test easy accuracy: {:.3f}".format(task, args.model, num_correct / num_tests))
    
    num_correct, num_tests = 0, 0
    test_sampler = SubsetRandomSampler(torch.arange(len(test_set_hard)))
    test_dataloader = GraphDataLoader(
        test_set_hard, sampler=test_sampler, batch_size=32, drop_last=False
    )
    for batched_graph, labels in test_dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        e_feat = None
        if 'feat' in batched_graph.edata:
            e_feat = batched_graph.edata["feat"].float()
        pred = model(batched_graph, batched_graph.ndata["feat"].float(), e_feat)
        if out_feat == 1:
            pred = torch.round(pred)
            num_correct += (pred.squeeze(-1) == labels).sum().item()
        else:
            num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)
    with open('GNN_results.txt', '+a') as f:
        f.write("Task: {}, Model: {}, Test hard accuracy: {:.3f}\n".format(task, args.model, num_correct / num_tests))
    print("Task: {}, Model: {}, Test hard accuracy: {:.3f}".format(task, args.model, num_correct / num_tests))