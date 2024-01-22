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
        self._use_async = False
        self._recv_shape, self._send_shape = [], []
        self._select_recv_num, self._select_send_num = [], []
        # embedding / gradient recv in gpu
        self._f_buf = []
        self._f_recv, self._b_recv = [], []
        # embedding / gradient send/recv in cpu
        self._f_send_cpu, self._f_recv_cpu = [], []
        self._g_send_cpu, self._g_recv_cpu = [], []

        #sample
        self._selected_send_mask, self._selected_recv_mask = [], []
        self._selected_send_mask_cpu, self._selected_recv_mask_cpu = [], []
    
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
        self._sample_rate = sample_rate
        self._backend = backend
        self._use_async = use_async
        self._stale_t = stale_t # indicate how many step async used

        self._send_shape = [None] * size
        self._select_recv_num, self._select_send_num = [None] * size, [None] * size
        for i in range(size):
            if i ==rank:
                continue
            self._send_shape[i] = boundary[i].shape[0]
            self._select_send_num[i] = int(self._send_shape[i] *  sample_rate)
            self._select_recv_num[i] = int(self._recv_shape[i] *  sample_rate)

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
                    s1 = torch.Size([self._recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([self._send_shape[j], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda'))
                    tmp2.append(torch.zeros(s2, device='cuda'))
            self._f_recv[i] = tmp1
            if i > 0:
                self._b_recv[i] = tmp2

        if backend == 'gloo':
            self._f_send_cpu, self._f_recv_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._g_send_cpu, self._g_recv_cpu = [None] * self._n_layers, [None] * self._n_layers

            for i in range(self._n_layers):
                tmp1, tmp2, tmp3, tmp4 = [], [], [], []
                for j in range(size):
                    if j == rank:
                        tmp1.append(None)
                        tmp2.append(None)
                        tmp3.append(None)
                        tmp4.append(None)
                    else:
                        s1 = torch.Size([self._select_send_num[j], self._layer_size[i]])
                        s2 = torch.Size([self._select_recv_num[j], self._layer_size[i]])
                        tmp1.append(torch.zeros(s1).pin_memory())
                        tmp2.append(torch.zeros(s2).pin_memory())
                        tmp3.append(torch.zeros(s2).pin_memory())
                        tmp4.append(torch.zeros(s1).pin_memory())
                self._f_send_cpu[i] = tmp1
                self._f_recv_cpu[i] = tmp3
                if i > 0:
                    self._g_send_cpu[i] = tmp2
                    self._g_recv_cpu[i] = tmp4

        self._selected_send_mask, self._selected_recv_mask = [None] * size, [None] * size
        self._selected_send_mask_cpu, self._selected_recv_mask_cpu = [None] * size, [None] * size
        for i in range(size):
            if rank == i:
                    continue
            temp1 = torch.zeros(self._send_shape[i], dtype=torch.bool, device='cuda')
            temp2 = torch.zeros(self._recv_shape[i], dtype=torch.bool, device='cuda')
            temp3 = torch.zeros(self._send_shape[i], dtype=torch.bool, pin_memory=True)
            temp4 = torch.zeros(self._recv_shape[i], dtype=torch.bool, pin_memory=True)
            self._selected_send_mask[i] = temp1
            self._selected_recv_mask[i] = temp2
            self._selected_send_mask_cpu[i] = temp3
            self._selected_recv_mask_cpu[i] = temp4
        

        self.__init_pl_pr()

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
            for i in range(size):
                if i == rank:
                    continue
                idx = torch.as_tensor(np.random.choice(self._recv_shape[i], self._select_recv_num[i], replace=False),dtype=torch.long)
                self._selected_recv_mask_cpu[i][idx] = True
        elif smple_method == "degree":
            pass
        elif smple_method == "":
            pass

        req1, req2 = queue.Queue(), queue.Queue()
        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size
            r2 = dist.irecv(self._selected_send_mask_cpu[left], src=left)
            req2.put((r2, left))
            r1 = dist.isend(self._selected_recv_mask_cpu[right], dst=right)
            req1.put((r1, right))

        while not req2.empty():
            r, rank_inx = req2.get()
            r.wait()
            self._selected_send_mask[rank_inx].copy_(self._selected_send_mask_cpu[rank_inx])
            self._selected_send_mask_cpu[rank_inx].zero_()
        while not req1.empty():
            r, rank_inx = req1.get()
            self._selected_recv_mask[rank_inx].copy_(self._selected_recv_mask_cpu[rank_inx])
            r.wait()
            self._selected_recv_mask_cpu[rank_inx].zero_()

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

    def __gloo_all_to_all(self, send_gpu, recv_gpu, send_cpu, recv_cpu, tag, forward = True):
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()
                
        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size

            # prepare send tensor
            if forward:     
                send_cpu[right].copy_(send_gpu[self._boundary[right]][self._selected_send_mask[right]])
            else:
                send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]][self._selected_recv_mask[right]])

            r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
            req2.put((r2, left))
            r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
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
            self.__gloo_all_to_all(feat, self._f_recv[layer], self._f_send_cpu[layer], self._f_recv_cpu[layer], tag, forward = True)
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
            self.__gloo_all_to_all(grad, self._b_recv[layer], self._g_send_cpu[layer], self._g_recv_cpu[layer], tag, forward = False)
        else:
            raise NotImplementedError