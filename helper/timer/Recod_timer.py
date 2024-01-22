import time
import torch.distributed as dist
from contextlib import contextmanager

class RecodTimer(object):

    def __init__(self, name = ""):
        super(RecodTimer, self).__init__()
        self._time = []
        self._name = name

    @contextmanager
    def timer(self):
        t0 = time.time()
        yield
        t1 = time.time()
        self._time.append(t1-t0)

    def tot_time(self):
        tot = 0
        for t in self._time:
            tot += t
        return tot

    def print_time(self):
        rank, size = dist.get_rank(), dist.get_world_size()
        for (k, t) in self._time.items():
            print(f'(rank {rank}) {self._name} time of {k}: {t} seconds.')
    def rocord_time(self):
        import datetime
        rank, size = dist.get_rank(), dist.get_world_size()
        file_name = "results/" + self._name + "--" + str(rank) + "--" + str(datetime.datetime.now())  + ".txt"
        with open(file_name, "w") as f:
            for t in self._time:
                f.write(str(t)+"\n")
    def clear(self):
        self._time = []
