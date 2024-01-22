import argparse
import torch
import dgl
import os
from dgl.data import RedditDataset,CoraGraphDataset,YelpDataset
import scipy
import numpy as np
from sklearn.preprocessing import StandardScaler
from ogb.nodeproppred import DglNodePropPredDataset
import json

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

# def load_yelp():
#     prefix = './dataset/yelp/'

#     with open(prefix + 'class_map.json') as f:
#         class_map = json.load(f)
#     with open(prefix + 'role.json') as f:
#         role = json.load(f)

#     adj_full = scipy.sparse.load_npz(prefix + 'adj_full.npz')
#     feats = np.load(prefix + 'feats.npy')
#     n_node = feats.shape[0]

#     g = dgl.from_scipy(adj_full)
#     node_data = g.ndata

#     label = list(class_map.values())
#     node_data['label'] = torch.tensor(label)

#     node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
#     node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
#     node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
#     node_data['train_mask'][role['tr']] = True
#     node_data['val_mask'][role['va']] = True
#     node_data['test_mask'][role['te']] = True

#     assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['val_mask'])))
#     assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['test_mask'])))
#     assert torch.all(torch.logical_not(torch.logical_and(node_data['val_mask'], node_data['test_mask'])))
#     assert torch.all(
#         torch.logical_or(torch.logical_or(node_data['train_mask'], node_data['val_mask']), node_data['test_mask']))

#     train_feats = feats[node_data['train_mask']]
#     scaler = StandardScaler()
#     scaler.fit(train_feats)
#     feats = scaler.transform(feats)
#     node_data['feat'] = torch.tensor(feats, dtype=torch.float)

#     return g


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
        # TODO: remove the following three lines later 
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
    return g

def graph_partition(g, args):
    graph_dir = 'partitions/' + args.graph_name + '/'
    part_config = graph_dir + args.graph_name + '.json'

    # TODO: after being saved, a bool tensor becomes a uint8 tensor (including 'inner_node')
    if not os.path.exists(part_config):
        with g.local_scope():  #使用local_scope() 范围时，任何对节点或边的修改在脱离这个局部范围后将不会影响图中的原始特征值 。
            if args.inductive: #对于inductive 训练的时候用不到测试集
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask') #  为什么对于inductive训练 在图划分前要pop出这两部分
            g.ndata['in_degree'] = g.in_degrees()
            dgl.distributed.partition_graph(g, args.graph_name, args.n_partitions, graph_dir, part_method=args.partition_method,
                             balance_edges=False, objtype=args.partition_obj)
            
def downloadAndPartitionGraph(args):
    g = load_data(args.dataset)
    
    if args.inductive:
        args.graph_name = '%s-%d-%s-%s-induc' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)
        graph_partition(g.subgraph(g.ndata['train_mask']), args)
    else:
        args.graph_name = '%s-%d-%s-%s-trans' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)
        graph_partition(g, args)
    
    # 原full graph的n_feat、n_class、n_train
    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]
    n_train = g.ndata['train_mask'].int().sum().item()
    # 打开partitions/args.dataset/args.graph_name/args.graph_name.json
    # 将 nfeat、n_class、n_train 添加到json文件中
    part_config = 'partitions/' + args.graph_name + '/' + args.graph_name + '.json'
    with open(part_config, 'r+') as conf_f:
        part_metadata = json.load(conf_f)

        part_metadata['n_feat'] = n_feat
        part_metadata['n_class'] = n_class
        part_metadata['n_train'] = n_train

        conf_f.seek(0)
        json.dump(part_metadata, conf_f,sort_keys=True, indent=4)
        conf_f.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download and Partition Graph')
    parser.add_argument("--dataset", type=str, default='reddit', help="the input dataset")
    parser.add_argument("--inductive", action='store_true', help="inductive learning setting")
    parser.add_argument("--n-partitions", "--n_partitions", type=int, default=2, help="the number of partitions")
    parser.add_argument("--partition-obj", "--partition_obj", choices=['vol', 'cut'], default='vol', help="partition objective function ('vol' or 'cut')")
    parser.add_argument("--partition-method", "--partition_method", choices=['metis', 'random'], default='metis', help="the method for graph partition ('metis' or 'random')")
    args = parser.parse_args()
    downloadAndPartitionGraph(args)