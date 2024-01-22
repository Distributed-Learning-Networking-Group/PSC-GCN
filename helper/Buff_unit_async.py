import torch
from multiprocessing import Event

class Buff_Unit(object):
    def __init__(self, send_shape, recv_shape, n_layers, layer_size, rank, size, backend) -> None:
        super(Buff_Unit, self).__init__()

        # cpu f/g send/recv
        self.f_send_cpu, self.g_send_cpu = [None] * n_layers, [None] * n_layers
        self.f_recv_cpu, self.g_recv_cpu = [None] * n_layers, [None] * n_layers
        # gpu f/g recv
        self.f_recv_gpu, self.g_recv_gpu = [None] * n_layers, [None] * n_layers
        # event
        self.f_cpu_event, self.b_cpu_event = [None] * n_layers, [None] * n_layers
        self.f_cuda_event, self.b_cuda_event = [None] * n_layers, [None] * n_layers

        # f/g  send/recv in cpu memory
        if backend == 'gloo':
            for i in range(n_layers):
                tmp1, tmp2, tmp3, tmp4 = [], [], [], []
                for j in range(size):
                    if j == rank:
                        tmp1.append(None)
                        tmp2.append(None)
                        tmp3.append(None)
                        tmp4.append(None)
                    else:
                        s1 = torch.Size([send_shape[j], layer_size[i]])
                        s2 = torch.Size([recv_shape[j], layer_size[i]])
                        tmp1.append(torch.zeros(s1).pin_memory())
                        tmp2.append(torch.zeros(s2).pin_memory())
                        tmp3.append(torch.zeros(s2).pin_memory())
                        tmp4.append(torch.zeros(s1).pin_memory())
                self.f_send_cpu[i] = tmp1
                self.f_recv_cpu[i] = tmp3
                if i > 0:
                    self.g_send_cpu[i] = tmp2
                    self.g_recv_cpu[i] = tmp4
        
        # f/g recv in gpu
        for i in range(n_layers):
            tmp1, tmp2, tmp3, tmp4 = [], [], [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                else:
                    s1 = torch.Size([recv_shape[j], layer_size[i]])
                    s2 = torch.Size([send_shape[j], layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda'))
                    tmp2.append(torch.zeros(s2, device='cuda'))
            self.f_recv_gpu[i] = tmp1
            if i > 0:
                self.g_recv_gpu[i] = tmp2

            self.f_cpu_event[i] = Event()
            self.b_cpu_event[i] = Event()
            self.f_cuda_event[i] = torch.cuda.Event()
            self.b_cuda_event[i] = torch.cuda.Event()
