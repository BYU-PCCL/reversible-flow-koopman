import colorama
from collections import defaultdict

class block(object):
    def __init__(self, header, size=80):
        pre = size - len(header)
        pre //= 2
        post = pre
        print('{} {} {}'.format('=' * (pre - 1), header, '=' * (post - 1)))

    def print(self, string, indent=True):
        print('{}{}'.format(' ' if indent else '', string))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
      print()

class LazyOptimizer():
    def __init__(self, optimizer, model):
        self.__dict__.update(locals())
    
    def zero_grad(self):
        if self._initialized:
            self.optimizer.zero_grad()

    def backward(self):
        self.optimizer

def gpustats():
    import py3nvml.py3nvml as pynvml

    if '__gpuhandler__' not in globals():
        globals()['__gpuhandler__'] = True
        pynvml.nvmlInit()

    usage = []
    util = []
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage.append(info.used / info.total)
        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        util.append(info.gpu / 100.)

    return {'maxmemusage': max(usage), 'maxutil': max(util)}