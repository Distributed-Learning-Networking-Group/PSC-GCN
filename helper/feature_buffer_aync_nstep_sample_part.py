import torch
from multiprocessing.pool import ThreadPool
from helper.timer.timer import *
from helper.Buff_unit_async_sample import *
from helper.cache import Cache
import queue
import numpy as np
import dgl.function as fn

class Buffer(object):

    def __init__(self):
        super(Buffer, self).__init__()
        self._rank = None
        self._size = None
        self._num_in = None
        self._num_all = None
        self._boundary = []
        self._n_layers = 0
        self._layer_size = []
        self._use_async = False
        self._use_sample = False
        self._use_cache = False
        self._cache_size = None
        self._epoch = 0
        self._buff = []
        self._embed_buf = []
        self._embed_recv, self._grad_recv = [], []
        self._recv_shape, self._send_shape = [], []
        self.cache = None
        self._pool = None
        self._step = None
        self._backend = None
        self._pl, self._pr = [], []
        self._comm_stream = []

    def __init_pl_pr(self):
        self._pl, self._pr = [], []
        tot = self._num_in
        for s in self._recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)

    def init_buffer(self, graph, feat, num_in, num_all, boundary, f_recv_shape, layer_size, 
                    use_sample = False, sample_rate =1, sample_method = 'random', recompute_every = 100,
                    use_async=False, stale_t= 1, use_cache = False, cache_size = 0, cache_policy = 'random', backend='gloo'):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._rank = rank
        self._size = size
        self._num_in = num_in
        self._num_all = num_all
        self._boundary = boundary
        self._n_layers = len(layer_size)
        self._layer_size = layer_size
        self._use_sample = use_sample
        self._use_async = use_async
        self._use_cache = use_cache
        self._cache_size = cache_size
        self._epoch = 0
        self._recv_shape = f_recv_shape
        self._sample_rate = sample_rate
        self._sample_method = sample_method
        self._recompute_every = recompute_every
        self._stale_t = stale_t # indicate how many step async used
        self._step = 0 #indicate which step in
        self._backend = backend
        if use_sample:
            self.sample_matrix = torch.zeros([graph.num_edges()],device='cuda') + self._sample_rate
            self.sample_matrix_temp = torch.zeros([graph.num_edges()],device='cuda') + self._sample_rate
        # stream
        for i in range(stale_t):
            self._comm_stream.append(torch.cuda.Stream())
        self._send_shape = [None] * size
        for i in range(size):
            if i ==rank:
                continue
            self._send_shape[i] = boundary[i].shape[0]

        self._embed_buf = [None] * self._n_layers
        self._embed_recv, self._grad_recv = [None] * self._n_layers, [None] * self._n_layers
        for i in range(self._n_layers):
            self._embed_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda', requires_grad=True)
            tmp1, tmp2 = [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                else:
                    s1 = torch.Size([self._recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([self._send_shape[j], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda', requires_grad=False))
                    tmp2.append(torch.zeros(s2, device='cuda', requires_grad=False))
            self._embed_recv[i] = tmp1
            if i > 0:
                self._grad_recv[i] = tmp2
        self.__init_pl_pr()
        if use_cache:
            self.cache = Cache(f_recv_shape, self._layer_size[0], cache_size=cache_size, rank=rank ,size=size, cache_policy=cache_policy)
            self.cache.initCache(feat, self._boundary, total_recv = self._num_all-self._num_in, pl=self._pl, pr = self._pr, plrdel = self._num_in, g=graph)
        if use_async:
            for i in range(self._stale_t):
                self._buff.append(Buff_Unit(self._send_shape, self._recv_shape, self._n_layers, self._layer_size, rank, size, backend, use_sample=use_sample, use_cache=use_cache, cache=self.cache))
                self._pool = ThreadPool(processes=2*self._n_layers*self._stale_t)
        else:
            self._buff.append(Buff_Unit(self._send_shape, self._recv_shape, self._n_layers, self._layer_size, rank, size, backend, use_sample=use_sample, use_cache=use_cache, cache=self.cache))
  
    @torch.no_grad()
    def sampleMatrixUpdate(self, graph, feat):
        t0 = time.time()
        for i in range(1,self._n_layers-1):
            if i == 1:
                y_ij = torch.norm(self._embed_buf[1], p=2, dim=1) 
            else:
                y_ij = y_ij + torch.norm(self._embed_buf[i], p=2, dim=1)
        node_id = graph.nodes('_V')
        inner_node = node_id.numel()
        device = node_id.device
        src, dst = graph.edges(order='eid')
        alpha = 1.1
        y_ij[:inner_node] = alpha * y_ij[:inner_node]
        e_data = y_ij[src.long()]
        indegree = graph.in_degrees()
        in_edges_dict = []
        
        for node in node_id:
            edge_id = graph.in_edges(node, form='eid', etype=None).long()
            sorted_values, sorted_indices = torch.sort(e_data[edge_id])
            sorted_sqrt_values = torch.sqrt(sorted_values)
            node_indegree = indegree[node].item()
            B = round(node_indegree * self._sample_rate)
            min_k = node_indegree - B
            max_k = node_indegree - B + 1
            cumsum = torch.cumsum(sorted_sqrt_values, dim=0)
            for k in range(node_indegree, min_k, -1):
                # sum_k = torch.sum(sorted_sqrt_values[:k])
                sum_k = cumsum[k-1]
                if (sum_k / (B+k-node_indegree)) > sorted_sqrt_values[k-1]:
                    max_k = k
                    break

            sum_max_k = torch.sum(sorted_sqrt_values[:max_k]).to(device)
            p = torch.zeros(node_indegree,device='cuda')
            p[:max_k] = ((B+max_k-node_indegree) * sorted_sqrt_values[:max_k]) / sum_max_k
            p[max_k:] = 1
            indices = edge_id[sorted_indices]
            self.sample_matrix_temp[indices] = p
        print("计算采样时长:",time.time() - t0)
        print(torch.mean(self.sample_matrix_temp))
    def setSampleMatrix(self):
        self.sample_matrix.copy_(self.sample_matrix_temp)
    def updateCache(self, graph, feat):
        if self._use_cache:
            recv_idx = [None] * self._size
            rmt_nodes = torch.arange(self._num_in,self._num_all,dtype=torch.int32,device='cuda')

            self.sample_matrix_temp = 1- self.sample_matrix_temp
            graph.edges['_E'].data['p'] = self.sample_matrix_temp
            b_edges = graph.out_edges(rmt_nodes, form='eid')
            sub_g = dgl.edge_subgraph(graph, {('_U', '_E', '_V'): b_edges} ,relabel_nodes=False,store_ids=False,output_device ='cuda')
            rg = dgl.reverse(sub_g, copy_ndata=False, copy_edata=True)
            def reducer(nodes):
                # print(nodes.mailbox['m'])
                return {'n': nodes.mailbox['m'].prod(1)}
            rg.update_all(fn.copy_e('p', 'm'), reducer,etype='_E')
            rg.nodes['_U'].data['n'] = 1 - rg.nodes['_U'].data['n']
            p = rg.nodes['_U'].data['n'][self._num_in:self._num_all]
            topk_values, topk_indices = torch.topk(p, self._cache_size)
            topk_indices, _ = topk_indices.sort()
            recv_idx = [None] * self._size
            for i in range(self._size):
                if i == self._rank:
                    continue
                recv_idx[i] =topk_indices[torch.nonzero((topk_indices>=(self._pl[i]-self._num_in)) == (topk_indices<(self._pr[i]-self._num_in)), as_tuple=True)[0]] - (self._pr[i]-self._num_in)
            print(f"cache update begin")
            self.cache.updateCache(recv_idx, feat, self._boundary)
            print(f"cache update end")
        
    def nextEpoch(self):
        self._epoch += 1
        self._step = (self._step + 1) % self._stale_t
    def setCommuInfo(self):
        if self._use_sample:
            self._buff[self._step].setCommuInfo()

    def initSampleNodes(self, original_graph, epoch):
        return self._buff[self._step].sampleNodes(original_graph,self.sample_matrix, self._sample_method, epoch, self._recompute_every, self._stale_t*int(self._use_async), self._pl, self._pr, self.cache)

    def sampleNodes(self, original_graph, epoch):
        next_step = (self._step + 1) % self._stale_t
        return self._buff[next_step].sampleNodes(original_graph, self.sample_matrix, self._sample_method, epoch, self._recompute_every, self._stale_t*int(self._use_async), self._pl, self._pr, self.cache)

    def __feat_fusion(self, layer, step):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i != rank:
                if layer == 0 and self._use_cache:
                    self._embed_recv[layer][i][self._buff[step].get_hit_recv_idx()[i]] = self.cache.fetch_from_cache(self._buff[step].get_cache_idx(),i)
                    self._embed_recv[layer][i][self._buff[step].get_miss_recv_idx()[i]] = self._buff[step].get_f_recv_gpu(layer)[i]
                else:
                    self._embed_recv[layer][i][self._buff[step].get_recv_idx()[i]] = self._buff[step].get_f_recv_gpu(layer)[i]

    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                tmp.append(self._embed_recv[layer][i])
        return torch.cat(tmp)

    def update(self, layer, feat):
        torch.cuda.current_stream().synchronize()
        if self._use_async is False:
            with comm_timer.timer(f'epoch:{self._epoch} forward_{layer}'):
                self.__feat_transfer(self._epoch, self._step, layer, feat)
                torch.cuda.current_stream().wait_event(self._buff[self._step].get_f_cuda_event(layer))
            # self.__feat_fusion(layer, self._step) #TODO in commu?
            self._embed_buf[layer] = self.__feat_concat(layer, feat)
            if self._embed_buf[layer].requires_grad:
                self._embed_buf[layer].register_hook(self.__grad_hook(self._epoch, layer, self._step))
            return self._embed_buf[layer]
        else:
            if self._epoch > self._stale_t - 1:
                with comm_timer.timer(f'epoch:{self._epoch} forward_{layer}'):
                    self._buff[self._step].get_f_cpu_event(layer).wait()
                    torch.cuda.current_stream().wait_event(self._buff[self._step].get_f_cuda_event(layer))
                    self._buff[self._step].get_f_cpu_event(layer).clear()  # Event.wait() will block
                # self.__feat_fusion(layer, self._step) # TODO in commu?
            self._embed_buf[layer] = self.__feat_concat(layer, feat)
            self._pool.apply_async(self.__feat_transfer, args=(self._epoch, self._step, layer, feat))
            if self._embed_buf[layer].requires_grad and layer > 0 :
                self._embed_buf[layer].register_hook(self.__grad_hook(self._epoch, layer, self._step))
            return self._embed_buf[layer]

    @torch.no_grad()
    def __gloo_all_to_all(self, send_gpu, recv_gpu, send_cpu, recv_cpu, selected_send_idx, selected_recv_idx, step, tag, forward = True):
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()
        self._comm_stream[step].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._comm_stream[step]):
            for i in range(1, size):
                left = (rank - i + size) % size
                right = (rank + i) % size
                r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
                req2.put((r2, left))
                # prepare send tensor
                if forward:     
                    send_cpu[right].copy_(send_gpu[self._boundary[right]][selected_send_idx[right]])
                else:
                    send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]][selected_recv_idx[right]])
                r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
                req1.append(r1)

            while not req2.empty():
                r, rank_inx = req2.get()
                # TODO: if r.is_completed() run following lines else next r (see issue #30723)
                r.wait()
                if forward:
                    recv_gpu[rank_inx].copy_(recv_cpu[rank_inx], non_blocking = True)
                else:
                    recv_gpu[rank_inx].copy_(recv_cpu[rank_inx], non_blocking = True)
            # TODO: remove this 'wait'
            for r in req1:
                r.wait()


    def __feat_transfer(self, epoch, step, layer, feat):
        tag = TransferTag.FEAT_BEGIN + epoch * 2 * self._n_layers + layer
        if self._backend == 'gloo':
            # with torch.no_grad():
            if layer == 0 and self._use_cache:
                self.__gloo_all_to_all(feat, self._buff[step].get_f_recv_gpu(layer), self._buff[step].get_f_send_cpu(layer), self._buff[step].get_f_recv_cpu(layer),
                                        self._buff[step].get_miss_send_idx(), self._buff[step].get_miss_recv_idx(), step, tag, forward=True)
            else:
                self.__gloo_all_to_all(feat, self._buff[step].get_f_recv_gpu(layer), self._buff[step].get_f_send_cpu(layer), self._buff[step].get_f_recv_cpu(layer),
                                        self._buff[step].get_send_idx(), self._buff[step].get_recv_idx(), step, tag, forward=True)
            self._buff[step].get_f_cuda_event(layer).record(self._comm_stream[step])
        else:
            raise NotImplementedError
        self.__feat_fusion(layer, step)
        self._buff[step].get_f_cpu_event(layer).set() # Event.wait() will not block
        #add at 7.24
        self._buff[step].update_f_event[layer].set()


    def __grad_fusion(self, layer, step):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i != rank:
                self._grad_recv[layer][i][self._buff[step].get_send_idx()[i]] = self._buff[step].get_g_recv_gpu(layer)[i]


    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i == rank:
                continue
            else:
                grad[self._boundary[i]] += self._grad_recv[layer][i]

    def __grad_hook(self, epoch, layer, step):
        def fn(grad):
            torch.cuda.current_stream().synchronize()
            if self._use_async is False:
                with comm_timer.timer(f'epoch:{epoch} backward_{layer}'):
                    self.__grad_transfer(epoch, step, layer, grad)
                    torch.cuda.current_stream().wait_event(self._buff[step].get_b_cuda_event(layer))
                # self.__grad_fusion(layer, step)
                self.__update_grad(layer, grad)
                return grad
            else:
                if self._epoch > self._stale_t - 1:
                    with comm_timer.timer(f'epoch:{epoch} backward_{layer}'):
                        self._buff[step].get_b_cpu_event(layer).wait()   # wait last eopch backward finish : self._b_cpu_event[layer].set()
                        torch.cuda.current_stream().wait_event(self._buff[step].get_b_cuda_event(layer))
                        self._buff[step].get_b_cpu_event(layer).clear()  # Event.wait() will not block
                    # self.__grad_fusion(layer, step)
                self.__update_grad(layer, grad)
                self._pool.apply_async(self.__grad_transfer, args=(epoch, step, layer, grad), error_callback=lambda x:print('error!!!'))
                return grad
        return fn

    def __grad_transfer(self, epoch, step, layer, grad):
        tag = TransferTag.FEAT_BEGIN + epoch * 2 * self._n_layers + layer + self._n_layers
        if self._backend == 'gloo':
            self.__gloo_all_to_all(grad, self._buff[step].get_g_recv_gpu(layer), self._buff[step].get_g_send_cpu(layer), self._buff[step].get_g_recv_cpu(layer),
                                    self._buff[step].get_send_idx(), self._buff[step].get_recv_idx(), step, tag, forward=False)
            self._buff[step].get_b_cuda_event(layer).record(self._comm_stream[step])
        else:
            raise NotImplementedError
        self.__grad_fusion(layer, step)
        self._buff[step].get_b_cpu_event(layer).set()  # Event.wait() will not block
        #add at 7.24
        self._buff[step].update_b_event[layer].set()
    
    def wait(self):
        for i in range(self._n_layers):
            self._buff[self._step].update_f_event[i].wait()
            self._buff[self._step].update_f_event[i].clear()
        for i in range(1,self._n_layers):
            self._buff[self._step].update_b_event[i].wait()
            self._buff[self._step].update_b_event[i].clear()
        #add at 7.24
