import torch
import torch.distributed as dist
from multiprocessing.pool import ThreadPool

# 在train.py: ctx.reducer.synchronize()
class Reducer(object):

    def __init__(self):
        super(Reducer, self).__init__()
        self._data_cpu = {}# dictionary：key:named_parameters的name，value:tuple(shape类似于param.data的0 tensor，新的group)
        self._pool = None #线程池共有named_parameters的个数的线程
        self._handles = []
        self._stream = None

    def init(self, model):
        cnt = 0
        for i, (name, param) in enumerate(model.named_parameters()):
            cnt += 1
            # group：
            # 即进程组。默认情况下，只有一个组，一个 job 即为一个组，也即一个 world。
            # 当需要进行更加精细的通信时，可以通过 new_group(ranks=None；list[所有在group中的rank]；, timeout=datetime.timedelta(seconds=1800), backend=None, pg_options=None) 接口，使用 word 的子集，创建新组，用于集体通信等。
            # 即所有进程中使用了dist.new_group()的形成一个新的组
            self._data_cpu[name] = (torch.zeros_like(param.data, pin_memory=True, device='cpu'), dist.new_group())
        self._pool = ThreadPool(processes=cnt)
        self._stream = torch.cuda.Stream()
    # param，name，grad，n_train
    def reduce(self, param, name, data, n_train):
        def create_stream():
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                data.div_(n_train)# 除去partition的数量
                data_cpu, group = self._data_cpu[name]
                data_cpu.copy_(data) # 将grad copy到cpu data_cpu中
                dist.all_reduce(data_cpu, op=dist.ReduceOp.SUM, group=group) # all——reduce
                param.grad.copy_(data_cpu, non_blocking=True) #更新gpu中的grad
        
        #apply_async：returns a AsyncResult object.
        self._handles.append(self._pool.apply_async(create_stream))
        
    # 同步all_reduce过程
    def synchronize(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
        torch.cuda.current_stream().wait_stream(self._stream)
