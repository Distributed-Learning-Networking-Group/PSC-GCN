import torch
import torch.distributed as dist
from helper.transfer_tag import *
import queue
import numpy as np
class Cache(object):
    """
    recv_num: 应该从其他part中获取的所有的node数量, 不经过sample
    
    """
    def __init__(self, recv_num, feat_dim, cache_size, rank, size, cache_policy= 'random') -> None:
        super(Cache, self).__init__()
        self._rank = rank
        self._size = size
        self._recv_num = recv_num
        self._cache_policy = cache_policy
        self._cache_size = cache_size
        self._cache_rate = None
        self._feat_dim = feat_dim
        self._feature_cache = [None] * size
        self._total_num = torch.tensor([0])
        self._hit_num = torch.tensor([0])
        self._cache_flag = [None] * size
        self._nodeidx2cacheidx = [None] * size
        for i in range(size):
            if i == rank:
                continue
            self._cache_flag[i] = torch.zeros(recv_num[i],device='cuda').bool()
            self._nodeidx2cacheidx[i] = torch.zeros(recv_num[i], dtype=torch.long, device='cuda')

    def initCache(self, feat, boundary, total_recv = 0, pl = [], pr = [], plrdel = 0, g = None):
        # print("plrdel ",plrdel)
        recv_idx = [None] * self._size
        if self._cache_policy == 'random' :
            idx = torch.as_tensor(np.random.choice(total_recv, self._cache_size, replace=False),dtype=torch.long, device='cuda')
            idx, _ = idx.sort()
            for i in range(self._size):
                if i == self._rank:
                    continue
                recv_idx[i] =idx[torch.nonzero((idx>=(pl[i]-plrdel)) == (idx<(pr[i]-plrdel)), as_tuple=True)[0]] -(pl[i] - plrdel)
            self.updateCache(recv_idx, feat, boundary)
        elif self._cache_policy == 'vr':
            g_outdeg = g.out_degrees(u='__ALL__', etype='_E')
            boundary_degree = g_outdeg[g.num_nodes('_V'):g.num_nodes('_U')]
            topk_values, topk_indices = torch.topk(boundary_degree, self._cache_size)
            topk_indices, _ = topk_indices.sort()
            for i in range(self._size):
                if i == self._rank:
                    continue
                recv_idx[i] =topk_indices[torch.nonzero((topk_indices>=(pl[i]-plrdel)) == (topk_indices<(pr[i]-plrdel)), as_tuple=True)[0]] -(pl[i] - plrdel)
            self.updateCache(recv_idx, feat, boundary)
        elif self._cache_policy == 'degree':
            g_outdeg = g.out_degrees(u='__ALL__', etype='_E')
            inner_outdeg = g_outdeg[:g.num_nodes('_V')]
            boundary_degree = torch.zeros(g.num_nodes('_U')-g.num_nodes('_V'), device='cuda')

            recv_boundary_deg_cpu = [None] * self._size
            send_cpu = [None] * self._size
            for i in range(self._size):
                if i == self._rank:
                    continue
                recv_boundary_deg_cpu[i] = torch.zeros(self._recv_num[i])
                send_cpu[i] = torch.zeros(boundary[i].shape[0])
            req1, req2 = [], queue.Queue()
            for i in range(1, self._size):
                left = (self._rank - i + self._size) % self._size
                right = (self._rank + i) % self._size

                r2 = dist.irecv(recv_boundary_deg_cpu[left], src=left, tag=TransferTag.CACHE_UPDATE3)
                req2.put((r2, left))
                send_cpu[right].copy_(inner_outdeg[boundary[right]])
                r1 = dist.isend(send_cpu[right], dst=right, tag=TransferTag.CACHE_UPDATE3)
                req1.append(r1)
            while not req2.empty():
                r, rank_inx = req2.get()
                r.wait()
            for r in req1:
                r.wait()
            tmp = []
            for i in range(self._size):
                if i != self._rank:
                    tmp.append(recv_boundary_deg_cpu[i])
            boundary_degree.copy_(torch.cat(tmp))

            topk_values, topk_indices = torch.topk(boundary_degree, self._cache_size)
            topk_indices, _ = topk_indices.sort()
            for i in range(self._size):
                if i == self._rank:
                    continue
                recv_idx[i] =topk_indices[torch.nonzero((topk_indices>=(pl[i]-plrdel)) == (topk_indices<(pr[i]-plrdel)), as_tuple=True)[0]] -(pl[i] - plrdel)
            self.updateCache(recv_idx, feat, boundary)
        
        

    def updateCache(self, recv_idx, feat, boundary):
        """
        update cache ,
        1.update through data transfer
        """
        # update cache flag
        for i in range(self._size):
            if i == self._rank:
                continue
            self._cache_flag[i].zero_()
            self._cache_flag[i][recv_idx[i]] = True
            self._nodeidx2cacheidx[i][recv_idx[i]] = torch.arange(int(torch.tensor(recv_idx[i].shape)), dtype=torch.long, device='cuda')

        recv_idx_cpu = [None] * self._size
        send_idx_cpu = [None] * self._size
        send_idx_gpu = [None] * self._size
        num_send = [None] * self._size
        # get send idx
        for i in range(1, self._size):
            left = (self._rank - i + self._size) % self._size
            right = (self._rank + i) % self._size
            num_send[left] = torch.tensor([0])
            recv_idx_cpu[right] = recv_idx[right].to('cpu')
            req = dist.isend(torch.tensor(recv_idx_cpu[right].shape), dst=right, tag=TransferTag.CACHE_UPDATE1)
            dist.recv(num_send[left], src=left, tag=TransferTag.CACHE_UPDATE1)

            send_idx_cpu[left] = torch.zeros(num_send[left], dtype=torch.long)
            send_idx_gpu[left] = torch.zeros(num_send[left], dtype=torch.long, device='cuda')
            req.wait()
            req = dist.isend(recv_idx_cpu[right], dst=right, tag=TransferTag.CACHE_UPDATE2)
            dist.recv(send_idx_cpu[left], src=left, tag=TransferTag.CACHE_UPDATE2)
            req.wait()
            send_idx_gpu[left].copy_(send_idx_cpu[left])
        
        # commu for feature in cache
        send_cpu = [None] * self._size
        recv_cpu= [None] * self._size
        for i in range(self._size):
            if i == self._rank:
                continue
            send_cpu[i] = torch.zeros([num_send[i], self._feat_dim])
            recv_cpu[i] = torch.zeros([int(torch.tensor(recv_idx_cpu[i].shape)), self._feat_dim])
            self._feature_cache[i] = torch.zeros([int(torch.tensor(recv_idx[i].shape)), self._feat_dim], device='cuda')
        req1, req2 = [], queue.Queue()
        for i in range(1, self._size):
            left = (self._rank - i + self._size) % self._size
            right = (self._rank + i) % self._size
            r2 = dist.irecv(recv_cpu[left], src=left, tag=TransferTag.CACHE_UPDATE3)
            req2.put((r2, left))
            send_cpu[right].copy_(feat[boundary[right]][send_idx_gpu[right]])
            r1 = dist.isend(send_cpu[right], dst=right, tag=TransferTag.CACHE_UPDATE3)
            req1.append(r1)
        while not req2.empty():
            r, rank_inx = req2.get()
            r.wait()
            self._feature_cache[rank_inx].copy_(recv_cpu[rank_inx], non_blocking = True)
        for r in req1:
            r.wait()


    def get_cache_idx(self, recv_node_idx):
        hit_idx = [None] * self._size
        cache_idx = [None] * self._size
        miss_idx = [None] * self._size
        for i in range(self._size):
            if i == self._rank:
                continue
            self._total_num += torch.tensor(recv_node_idx[i].shape)
            hit_idx[i] = torch.nonzero(self._cache_flag[i][recv_node_idx[i]], as_tuple=True)[0]
            miss_idx[i] = torch.nonzero(self._cache_flag[i][recv_node_idx[i]] == False, as_tuple=True)[0]
            cache_idx[i] = self._nodeidx2cacheidx[i][hit_idx[i]]
            self._hit_num += torch.tensor(hit_idx[i].shape)
            # torch.is_nonzero()
        return hit_idx, miss_idx, cache_idx

    def fetch_from_cache(self, cache_idx, rank_i):
        return self._feature_cache[rank_i][cache_idx[rank_i]]
    def get_miss_rate(self):
        return (self._total_num - self._hit_num) / self._total_num
    def get_hit_rate(self):
        return self._hit_num / self._total_num
    def get_hit_num(self):
        return self._hit_num
        
    def print_cache_info(self):
        print(f"|---------Cache in worker{self._rank}---------|")
        print(f"Cache total size: {self._cache_size}")
        for i in range(self._size):
            if i == self._rank :
                continue
            print(f"Cache flag for part {self._rank} : {self._cache_flag[i]}")
            print(f"Cache flag for part {self._rank} sum: {torch.sum(self._cache_flag[i])}")
            print(f"Cache buffer for part {self._rank} shape: {self._feature_cache[i].shape}")
