import torch
from multiprocessing.pool import ThreadPool
from helper.timer.timer import *
from helper.Buff_unit_async import Buff_Unit
import queue

class Buffer(object):

    def __init__(self):
        super(Buffer, self).__init__()
        self._num_in = None
        self._boundary = []
        self._n_layers = 0
        self._layer_size = []
        self._use_async = False
        self._epoch = 0
        self._buff = []
        self._embed_buf = []
        self._recv_shape = []
        self._send_shape = []
        self._pool = None
        self._step = None
        self._backend = None
        self._pl, self._pr = [], []
        # stream
        # self._comm_stream = None
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

    def init_buffer(self, num_in, num_all, boundary, f_recv_shape, layer_size, sample_rate =1, use_async=False, stale_t= 1, backend='gloo'):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._num_in = num_in
        self._boundary = boundary
        self._n_layers = len(layer_size)
        self._layer_size = layer_size
        self._use_async = use_async
        self._epoch = 0
        self._recv_shape = f_recv_shape
        self._sample_rate = sample_rate
        self._stale_t = stale_t # indicate how many step async used
        self._step = 0 #indicate which step in
        self._backend = backend
        # stream
        # self._comm_stream = torch.cuda.Stream()
        for i in range(stale_t):
            self._comm_stream.append(torch.cuda.Stream())

        self._send_shape = [None] * size
        for i in range(size):
            if i ==rank:
                continue
            self._send_shape[i] = boundary[i].shape[0]
        self._embed_buf = [None] * self._n_layers
        for i in range(self._n_layers):
            self._embed_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda')
        if use_async:
            for i in range(self._stale_t):
                self._buff.append(Buff_Unit(self._send_shape, self._recv_shape, self._n_layers, self._layer_size, rank, size, backend))
        else:
            self._buff.append(Buff_Unit(self._send_shape, self._recv_shape, self._n_layers, self._layer_size, rank, size, backend))
        self._pool = ThreadPool(processes=2*self._n_layers*self._stale_t)
        self.__init_pl_pr()

    def nextEpoch(self):
        self._epoch += 1
        self._step = (self._step + 1) % self._stale_t
    
    def selectNodes(self, smple_method):
        pass

    def __feat_concat(self, step, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                tmp.append(self._buff[step].f_recv_gpu[layer][i])
        return torch.cat(tmp)

    def update(self, layer, feat):
        torch.cuda.current_stream().synchronize()
        if self._use_async is False:
            with comm_timer.timer(f'epoch:{self._epoch} forward_{layer}'):
                self.__feat_transfer(self._epoch, self._step, layer, feat)
                torch.cuda.current_stream().wait_event(self._buff[self._step].f_cuda_event[layer])
            self._embed_buf[layer] = self.__feat_concat(self._step, layer, feat)
            if self._embed_buf[layer].requires_grad:
                self._embed_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
            return self._embed_buf[layer]
        else:
            if self._epoch > self._stale_t - 1:
                with comm_timer.timer(f'epoch:{self._epoch} forward_{layer}'):
                    self._buff[self._step].f_cpu_event[layer].wait()
                    torch.cuda.current_stream().wait_event(self._buff[self._step].f_cuda_event[layer])
                    self._buff[self._step].f_cpu_event[layer].clear()  # Event.wait() will block
            self._embed_buf[layer] = self.__feat_concat(self._step, layer, feat)
            self._pool.apply_async(self.__feat_transfer, args=(self._epoch, self._step, layer, feat))
            if self._embed_buf[layer].requires_grad:
                self._embed_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
            return self._embed_buf[layer]

    def __gloo_all_to_all(self, send_gpu, send_cpu, recv_cpu, recv_gpu, step, tag, forward=True):
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()
        self._comm_stream[step].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._comm_stream[step]):
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

    def __feat_transfer(self, epoch, step, layer, feat):
        tag = epoch * 2 * self._n_layers + layer
        if self._backend == 'gloo':
            self.__gloo_all_to_all(feat, self._buff[step].f_send_cpu[layer], self._buff[step].f_recv_cpu[layer],
                                    self._buff[step].f_recv_gpu[layer], step, tag, forward=True)
            self._buff[step].f_cuda_event[layer].record(self._comm_stream[step])
        else:
            raise NotImplementedError
        self._buff[step].f_cpu_event[layer].set() # Event.wait() will not block

    def __update_grad(self, layer, step, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i == rank:
                continue
            else:
                grad[self._boundary[i]] += self._buff[step].g_recv_gpu[layer][i]

    def __grad_hook(self, epoch, layer):
        def fn(grad):
            torch.cuda.current_stream().synchronize()
            if self._use_async is False:
                with comm_timer.timer(f'epoch:{self._epoch} backward_{layer}'):
                    self.__grad_transfer(epoch, self._step, layer, grad)
                    torch.cuda.current_stream().wait_event(self._buff[self._step].b_cuda_event[layer])
                self.__update_grad(layer, self._step, grad)
                return grad
            else:
                if self._epoch > self._stale_t - 1:
                    with comm_timer.timer(f'epoch:{self._epoch} backward_{layer}'):
                        self._buff[self._step].b_cpu_event[layer].wait()   # wait last eopch backward finish : self._b_cpu_event[layer].set()
                        torch.cuda.current_stream().wait_event(self._buff[self._step].b_cuda_event[layer])
                        self._buff[self._step].b_cpu_event[layer].clear()  # Event.wait() will not block
                self.__update_grad(layer, self._step, grad)
                self._pool.apply_async(self.__grad_transfer, args=(epoch, self._step, layer, grad))
                return grad
        return fn

    def __grad_transfer(self, epoch, step, layer, grad):
        tag = epoch * 2 * self._n_layers + layer + self._n_layers
        if self._backend == 'gloo':
            self.__gloo_all_to_all(grad, self._buff[step].g_send_cpu[layer], self._buff[step].g_recv_cpu[layer],
                                    self._buff[step].g_recv_gpu[layer], step, tag, forward=False)
            self._buff[step].b_cuda_event[layer].record(self._comm_stream[step])
        else:
            raise NotImplementedError
        self._buff[step].b_cpu_event[layer].set()  # Event.wait() will not block
