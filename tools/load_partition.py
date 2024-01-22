import argparse
import torch as th
import os
import torch.distributed as dist
import queue
import time
import dgl


def main(args):

    t0 = time.time()

    # TODO 1.load partition

    part_config = args.part_config


    print('loading partitions')
    """ 
    input:
    part_config - The path of the partition config file.
    part_id - The partition ID
    load_feats - Whether to load node/edge feats. If False, the returned node/edge feature dictionaries will be empty. Default: True.
    output:
    DGLGraph - The graph partition structure.
    Dict[str, Tensor] - Node features.
    Dict[(str, str, str), Tensor] - Edge features.
    GraphPartitionBook - The graph partition information.
    str - The graph name
    List[str] - The node types
    List[(str, str, str)] - The edge types
    """
    g, node_feat, edge_feat, gpb, graph_name, node_types, edge_types = dgl.distributed.load_partition( part_config=part_config, part_id=args.part_id)
    # gpb ，GraphPartitionBook ：The graph partition information.
    # gpb 是一个类，记录有图划分信息 
    #   有几个partition
    #   node或edge属于的partition ID
    #   a partition中拥有的 node IDs 和edge IDs
    #   在一个partition中nodes或edges的local IDs
    #可以用于local 、global  、partition IDs之间的转换
    print(g, node_feat, edge_feat, gpb, graph_name, node_types, edge_types)

    # TODO 1.load partition
    print(f"Loading prtition: {time.time() - t0}s\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loading partition")
    #dataset
    parser.add_argument("--dataset", type=str, default='reddit', help="the input dataset")
    #partition & location
    parser.add_argument("--graph_dir", type=str, default='', help="the input dataset")
    parser.add_argument("--part_config", type=str, default='', help="part_config - .json file")
    parser.add_argument("--part_id", type=int, default='0', help="number id of the partition will be loadeds")
    args = parser.parse_args()
    main(args)