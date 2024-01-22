import os
import scipy
import torch
import dgl
from dgl.data import RedditDataset, CoraGraphDataset, YelpDataset
from dgl.distributed import partition_graph
import torch.distributed as dist
import time
from contextlib import contextmanager
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from module.model import *
import torch.nn.functional as F
import netifaces

# %% load partition
def load_ogb_dataset(name):
    dataset = DglNodePropPredDataset(name=name, root='./dataset/')
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    n_node = g.num_nodes()
    node_data = g.ndata
    node_data['label'] = label.view(-1).long()
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True
    return g



def load_data(dataset):
    if dataset == 'reddit':
        data = RedditDataset(raw_dir='./dataset/')
        g = data[0]
    elif dataset == 'yelp':
        # g = load_yelp()
        dataset = YelpDataset(raw_dir='./dataset/')
        g = dataset[0]
        # add at 7.25
        g = g.long()
        g.ndata['label'] = g.ndata['label'].float()
        # TODO: remove the following three lines later (see Issue #4806 of DGL).
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()
        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
        # add at 7.25
    elif dataset == "cora":
        dataset = CoraGraphDataset(raw_dir='./dataset/')
        g = dataset[0]
    elif dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products')
    elif dataset == 'ogbn-papers100m':
        g = load_ogb_dataset('ogbn-papers100M')
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]
    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, n_feat, n_class


# %%get necessary data

def get_boundary(node_dict, gpb):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size  
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long)
        else:
            u = torch.zeros(num_left, dtype=torch.long, device=device)
        req.wait()
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.cuda()
        else:
            boundary[left] = u
        req.wait()
    return boundary

def get_layer_size(n_feat, n_class, n_hidden, n_layers):
    layer_size = [n_feat]
    layer_size.extend([n_hidden] * (n_layers - 1))
    layer_size.append(n_class)
    return layer_size
def get_pos(node_dict, gpb):
    pos = []
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i == rank:
            pos.append(None)
        else:
            part_size = gpb.partid2nids(i).shape[0]  
            start = gpb.partid2nids(i)[0].item()
            p = minus_one_tensor(part_size, 'cuda')
            in_idx = nonzero_idx(node_dict['part_id'] == i)
            out_idx = node_dict[dgl.NID][in_idx] - start
            p[out_idx] = in_idx
            pos.append(p)
    return pos

def get_recv_shape(node_dict): 
    rank, size = dist.get_rank(), dist.get_world_size()
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
        else:
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape



# %% setting before training
def create_inner_graph(graph, node_dict):
    u, v = graph.edges()
    sel = torch.logical_and(node_dict['inner_node'].bool()[u], node_dict['inner_node'].bool()[v])
    u, v = u[sel], v[sel]
    return dgl.graph((u, v))

def move_to_cuda(graph, part, node_dict):

    for key in node_dict.keys():
        node_dict[key] = node_dict[key].cuda()
    graph = graph.int().to(torch.device('cuda'))
    part = part.int().to(torch.device('cuda'))

    return graph, part, node_dict

def construct(part, graph, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    tot = part.num_nodes()
    u, v = part.edges()
    u_list, v_list = [u], [v]

    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]

            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            tot += u.shape[0]
            _, v_ = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v_)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')
    return g

def order_graph(part, graph, gpb, node_dict, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    one_hops = []
    for i in range(size):
        if i == rank:
            one_hops.append(None)
            continue
        start = gpb.partid2nids(i)[0].item()
        nodes = node_dict[dgl.NID][node_dict['part_id'] == i] - start
        nodes, _ = torch.sort(nodes)
        one_hops.append(nodes)

    return construct(part, graph, pos, one_hops)

def move_train_first(graph, node_dict, boundary):
    train_mask = node_dict['train_mask']
    num_train = torch.count_nonzero(train_mask).item()
    num_tot = graph.num_nodes('_V')

    new_id = torch.zeros(num_tot, dtype=torch.int, device='cuda')
    new_id[train_mask] = torch.arange(num_train, dtype=torch.int, device='cuda')
    new_id[torch.logical_not(train_mask)] = torch.arange(num_train, num_tot, dtype=torch.int, device='cuda')

    u, v = graph.edges()
    u[u < num_tot] = new_id[u[u < num_tot].long()]
    v = new_id[v.long()]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    for key in node_dict:
        node_dict[key][new_id.long()] = node_dict[key][0:num_tot].clone()
    for i in range(len(boundary)):
        if boundary[i] is not None:
            boundary[i] = new_id[boundary[i]].long()
    return graph, node_dict, boundary

def create_model(layer_size, n_train, args):
    if args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, norm=args.norm, dropout=args.dropout,
                         n_linear=args.n_linear, train_size=n_train)
    else:
        raise NotImplementedError

def reduce_hook(param, name, n_train):
    def fn(grad):
        ctx.reducer.reduce(param, name, grad, n_train)
    return fn


# %%training


def inductive_split(g):
    g_train = g.subgraph(g.ndata['train_mask'])
    g_val = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    g_test = g
    return g_train, g_val, g_test

def minus_one_tensor(size, device=None):
    if device is not None:
        return torch.zeros(size, dtype=torch.long, device=device) - 1
    else:
        return torch.zeros(size, dtype=torch.long) - 1

# b=tensor([ True, False,  True,  True,  True])
# nonzero_idx(b) = tensor([0, 2, 3, 4])
def nonzero_idx(x):
    return torch.nonzero(x, as_tuple=True)[0]

def print_memory(s):
    torch.cuda.synchronize()
    print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024
    ))

# %% validation & test

def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')

@torch.no_grad()           
def evaluate_induc(args, name, model, g, mode, test_accuray_rc, result_file_name=None):

    """
    mode: 'val' or 'test'
    """
    model.eval()  
    model.cpu()

    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
    test_accuray_rc.append(acc)
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, acc

@torch.no_grad()
def evaluate_trans(args, name, model, g, test_accuray_rc, result_file_name=None):
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    logits = model(g, feat)
    val_logits, test_logits = logits[val_mask], logits[test_mask]
    val_labels, test_labels = labels[val_mask], labels[test_mask]
    val_acc = calc_acc(val_logits, val_labels)
    test_acc = calc_acc(test_logits, test_labels)
    test_accuray_rc.append(test_acc)
    buf = "{:s} | Validation Accuracy {:.2%} | Test Accuracy {:.2%}".format(name, val_acc, test_acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, val_acc
# %% load partition graph

def load_partition(part_config, part_id, inductive, dataset):
    print('loading partitions')
    t0 = time.time()
    g, node_dict, edge_feat, gpb, graph_name, node_type, edge_types = dgl.distributed.load_partition(part_config=part_config, part_id = part_id)
    node_type = node_type[0]
    node_dict[dgl.NID] = g.ndata[dgl.NID]
    if 'part_id' in g.ndata:
        node_dict['part_id'] = g.ndata['part_id']
    node_dict['inner_node'] = g.ndata['inner_node'].bool()
    node_dict['label'] = node_dict[node_type + '/label']
    node_dict['feat'] = node_dict[node_type + '/feat']
    node_dict['in_degree'] = node_dict[node_type + '/in_degree']
    node_dict['train_mask'] = node_dict[node_type + '/train_mask'].bool()
    node_dict.pop(node_type + '/label')
    node_dict.pop(node_type + '/feat')
    node_dict.pop(node_type + '/in_degree')
    node_dict.pop(node_type + '/train_mask')
    if not inductive:
        node_dict['val_mask'] = node_dict[node_type + '/val_mask'].bool()
        node_dict['test_mask'] = node_dict[node_type + '/test_mask'].bool()
        node_dict.pop(node_type + '/val_mask')
        node_dict.pop(node_type + '/test_mask')
    if dataset == 'ogbn-papers100m':
        node_dict.pop(node_type + '/year')
    g.ndata.clear()
    g.edata.clear()
    print(f"Loading prtition: {time.time() - t0}s\n")
    return g, node_dict, gpb

# %% other function

# TODO assert parameters
def assertParameters(args):
    assert  args.sample_rate > 0 and args.sample_rate <= 1 , "sample_rate must in (0,1]"
    
def getNicName(args):
    if args.nic_name != "":
        return args.nic_name
    else:
        return netifaces.gateways()['default'][netifaces.AF_INET][1]
    
def reocord_time(args, train_dur, comm_dur, reduce_dur, loss_rc, test_accuray_rc):
    result_file_name = 'results/%s.txt' % (args.dataset)

    with open(result_file_name, 'a+') as f:
        f.write("This is rank:{:03d}".format(args.rank) + '\n')
        f.write("accuracy recod:" + str(test_accuray_rc) + '\n')
        f.write("loss recod:" + str(loss_rc) + '\n')
        f.write("train_dur:" + str((train_dur)) + '\n')
        f.write("comm_dur:" + str((comm_dur)) + '\n')
        f.write("reduce_dur:" + str((reduce_dur)) + '\n')
        f.close()
    print("|---------------Train time--------------|")
    print(f'max cost:{max(train_dur)}s')
    print(f'min cost:{min(train_dur)}s')
    print(f'average cost:{sum(train_dur) / len(train_dur)}s')
    print("|-----------communicate time------------|")
    print(f'max cost:{max(comm_dur)}s')
    print(f'min cost:{min(comm_dur)}s')
    print(f'average cost:{sum(comm_dur) / len(comm_dur)}s')
    print("|--------------reduce time--------------|")
    print(f'max cost:{max(reduce_dur)}s')
    print(f'min cost:{min(reduce_dur)}s')
    print(f'average cost:{sum(reduce_dur) / len(reduce_dur)}s')

