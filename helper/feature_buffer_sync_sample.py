import torch 
from helper.timer.timer import *
import queue
import numpy as np

class Buffer(object):

    def __init__(self):
        super(Buffer, self).__init__()
        self._rank = None
        self._size = None
        self._num_in = None
        self._boundary = []
        self._n_layers = 0 
        self._layer_size = []
        self._epoch = 0
        self._backend = None
        # embedding / gradient recv in gpu
        self._f_buf = []
        self._f_recv, self._b_recv = [], []
        self._recv_shape = []
        #sample
        self._select = None 
        self._selected_send_mask, self._selected_recv_mask = [], []
    
    def init_buffer(self, num_in, num_all, boundary, f_recv_shape, layer_size, sample_rate =1, use_async=False, stale_t= 1, backend='gloo'):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._rank = rank
        self._size = size
        self._num_in = num_in
        self._num_all = num_all
        self._boundary = boundary
        self._n_layers = len(layer_size) 
        self._layer_size = layer_size
        self._epoch = 0
        self._recv_shape = f_recv_shape
        self._backend = backend
        self._sample_rate = sample_rate
        self._use_async = use_async
        self._stale_t = stale_t # indicate how many step async used

        #initial buff in gpu
        self._f_buf = [None] * self._n_layers
        self._f_recv, self._b_recv = [], []
        self._f_recv, self._b_recv = [None] * self._n_layers, [None] * self._n_layers

        for i in range(self._n_layers):
            self._f_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda')
            tmp1, tmp2 = [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                else:
                    s1 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda'))
                    tmp2.append(torch.zeros(s2, device='cuda'))
            self._f_recv[i] = tmp1
            if i > 0:
                self._b_recv[i] = tmp2

        # init select mask buff
        self._select = torch.zeros(self._num_all - self._num_in, dtype=torch.bool).pin_memory()
        self._selected_send_mask, self._selected_recv_mask = [None] * size, [None] * size
        for i in range(size):
            if rank == i:
                    continue
            temp1 = torch.zeros(self._boundary[i].shape[0], dtype=torch.bool).pin_memory()
            temp2 = torch.zeros(f_recv_shape[i], dtype=torch.bool).pin_memory()
            self._selected_send_mask[i] = temp1
            self._selected_recv_mask[i] = temp2
        self.__init_pl_pr()

    def __init_pl_pr(self):
        self._pl, self._pr = [], []
        tot = 0
        for s in self._recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)    
    def nextEpoch(self):
        self._epoch += 1

    def selectNodes(self, smple_method):
        """
        when the process communicate with other processes, select partial neighbor nodes can efficiently reduce data volume, so then reduce
        communication time.
        there are three methods to select neighbor nodes, which represent the nodes in other partition but there is an edge pointing towards the nodes in this partition

        """
        rank, size = dist.get_rank(), dist.get_world_size()

        if smple_method == "random":
            select_num = int(self._sample_rate * (self._num_all - self._num_in))
            select_inx = torch.as_tensor(np.random.choice(self._num_all - self._num_in, select_num, replace=False), dtype=torch.long)
            self._select[select_inx] = True
            for i in range(size):
                self._selected_recv_mask[i] = self._select[self._pl[i]:self._pr[i]]
        elif smple_method == "degree":
            pass
        elif smple_method == "":
            pass

        req1, req2 = [], []
        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size

            r2 = dist.irecv(self._selected_send_mask[left], src=left)
            req2.append(r2)
            r1 = dist.isend(self._selected_recv_mask[right], dst=right)
            req1.append(r1)

        for r in req1:
            r.wait()
        for r in req2:
            r.wait()

        # self._selected_send_mask to cuda

    def getInfo(self):
        return None
    
    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                tmp.append(self._f_recv[layer][i])
        return torch.cat(tmp)

    def update(self, layer, feat):
        with comm_timer.timer(f'epoch:{self._epoch} forward_{layer}'):
            self.__feat_transfer(self._epoch, layer, feat)
        self._f_buf[layer] = self.__feat_concat(layer, feat)

        if self._f_buf[layer].requires_grad:
            self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
        return self._f_buf[layer]

    def __gloo_all_to_all(self, send_gpu, recv_gpu, layer, tag, forward = True):
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()
        recv_cpu = [None] * size
        for i in range(0, size):
            if i != rank:
                if forward:
                    recv_cpu_temp = torch.zeros([torch.sum(self._selected_recv_mask[i]), self._layer_size[layer]])
                else:
                    recv_cpu_temp = torch.zeros([torch.sum(self._selected_send_mask[i]), self._layer_size[layer]])
                recv_cpu[i] = recv_cpu_temp
                
        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size

            # prepare send tensor
            if forward:     
                send_cpu_temp = torch.zeros([torch.sum(self._selected_send_mask[right]), self._layer_size[layer]]) # TODO test .pin_memory()
                send_cpu_temp.copy_(send_gpu[self._boundary[right]][self._selected_send_mask[right]])
            else:
                send_cpu_temp = torch.zeros([torch.sum(self._selected_recv_mask[right]), self._layer_size[layer]]) # TODO test .pin_memory()
                send_cpu_temp.copy_(send_gpu[self._pl[right]+self._num_in:self._pr[right]+self._num_in][self._selected_recv_mask[right]])

            r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
            req2.put((r2, left))
            r1 = dist.isend(send_cpu_temp, tag=tag, dst=right)
            req1.append(r1)

        while not req2.empty():
            r, rank_inx = req2.get()
            # TODO: if r.is_completed() run following lines else next r (see issue #30723)
            r.wait()
            if forward:
                recv_gpu[rank_inx][self._selected_recv_mask[rank_inx]].copy_(recv_cpu[rank_inx], non_blocking = True)
            else:
                recv_gpu[rank_inx][self._selected_send_mask[rank_inx]].copy_(recv_cpu[rank_inx], non_blocking = True)
        # TODO: remove this 'wait'
        for r in req1:
            r.wait()


    def __feat_transfer(self, epoch, layer, feat):
        tag = epoch * 2 * self._n_layers + layer
        if self._backend == 'gloo':
            self.__gloo_all_to_all(feat, self._f_recv[layer], layer, tag, forward = True)
        else:
            raise NotImplementedError

    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):  
            if i == rank:
                continue
            else:
                grad[self._boundary[i]] += self._b_recv[layer][i]

    def __grad_hook(self, epoch, layer):
        def fn(grad):
            with comm_timer.timer(f'epoch:{self._epoch} backward_{layer}'):
                self.__grad_transfer(epoch, layer, grad)
            self.__update_grad(layer, grad)
            return grad
        return fn

    def __grad_transfer(self, epoch, layer, grad):
        tag = epoch * 2 * self._n_layers + layer + self._n_layers
        if self._backend == 'gloo':
            self.__gloo_all_to_all(grad, self._b_recv[layer], layer, tag, forward = False)
        else:
            raise NotImplementedError