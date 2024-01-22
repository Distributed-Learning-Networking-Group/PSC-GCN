import torch
from helper.timer.timer import *
import queue



class Buffer(object):

    def __init__(self):
        super(Buffer, self).__init__()
        self._num_in = None
        self._boundary = []
        self._n_layers = 0 
        self._layer_size = []
        self._epoch = 0

        self._feat_cpu, self._grad_cpu = [], []
        # use gpu
        self._f_buf = []
        # use gpu
        self._f_recv, self._b_recv = [], []
        self._f_recv_cpu, self._b_recv_cpu = [], []
        self._recv_shape = []
        self._backend = None
        self._pl, self._pr = [], []

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

    def init_buffer(self, num_in, num_all, boundary, f_recv_shape, layer_size, sample_rate =1, backend='gloo', use_async=False):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._num_in = num_in
        self._boundary = boundary
        self._n_layers = len(layer_size) 
        self._layer_size = layer_size
        self._epoch = 0
        self._recv_shape = f_recv_shape
        self._sample_rate = sample_rate
        self._backend = backend
        if backend == 'gloo':
            self._feat_cpu, self._grad_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._f_recv_cpu, self._b_recv_cpu = [None] * self._n_layers, [None] * self._n_layers
            for i in range(self._n_layers):

                # TODO：original feature 可以开始就存储 
                # if i == 0 :
                #     continue
                tmp1, tmp2, tmp3, tmp4 = [], [], [], []
                for j in range(size):
                    if j == rank:
                        tmp1.append(None)
                        tmp2.append(None)
                        tmp3.append(None)
                        tmp4.append(None)
                    else:
                        s1 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                        s2 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                        tmp1.append(torch.zeros(s1).pin_memory())
                        tmp2.append(torch.zeros(s2).pin_memory())
                        tmp3.append(torch.zeros(s2).pin_memory())
                        tmp4.append(torch.zeros(s1).pin_memory())
                self._feat_cpu[i] = tmp1
                self._f_recv_cpu[i] = tmp3

                if i > 0:
                    self._grad_cpu[i] = tmp2
                    self._b_recv_cpu[i] = tmp4
        # use gpu
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

        self.__init_pl_pr()

    def nextEpoch(self):
        self._epoch += 1
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
            self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))#注册一个 backward hook 每次 gradients 被计算的时候，这个 hook 都被调用。
        return self._f_buf[layer]
        
    def __gloo_all_to_all(self, send_gpu, send_cpu, recv_cpu, recv_gpu, tag, forward=True):
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()
        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size
            r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
            req2.put((r2, left))
            if forward:
                send_cpu[right].copy_(send_gpu[self._boundary[right]]) 
            else:
                send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]])
            r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
            req1.append(r1)

        while not req2.empty():
            r, idx = req2.get()
            # TODO: if r.is_completed() run following lines else next r (see issue #30723)
            r.wait()
            recv_gpu[idx].copy_(recv_cpu[idx], non_blocking=True)
        # TODO: remove this 'wait'
        for r in req1:
            r.wait()

    def __feat_transfer(self, epoch, layer, feat):
        tag = epoch * 2 * self._n_layers + layer
        if self._backend == 'gloo':
            self.__gloo_all_to_all(feat, self._feat_cpu[layer], self._f_recv_cpu[layer], self._f_recv[layer],tag, forward=True)   
        else:
            raise NotImplementedError

    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):  
            if i == rank:
                continue
            else:
                grad[self._boundary[i]] += self._b_recv[layer][i]  # gradient相加

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
            self.__gloo_all_to_all(grad, self._grad_cpu[layer], self._b_recv_cpu[layer], self._b_recv[layer],tag, forward=False)
        else:
            raise NotImplementedError
