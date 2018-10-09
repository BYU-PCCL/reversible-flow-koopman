import colorama
from collections import defaultdict, OrderedDict
import torch
import numpy as np
import builtins

is_profile = 'profile' in builtins.__dict__
profile = builtins.__dict__.get('profile', lambda x: x)

class block(object):
    def __init__(self, header, size=80, exit_on_error=False):
        self.exit_on_error = True
        pre = size - len(header)
        pre //= 2
        post = pre
        print('{} {} {}'.format('=' * (pre - 1), header, '=' * (post - 1)))

    def print(self, string, *args, indent=True):
        print('{}{}'.format(' ' if indent else '', string))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
      if type is not None:
        print('[{}] {}'.format(type.__name__, value))
        if self.exit_on_error:
            exit()
      print()

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

def pytorch_summary(module, verbose=False):
    if verbose:
        def __repr__(self):
            # We treat the extra repr like the sub-module, one item per line
            extra_lines = []
            extra_repr = self.extra_repr()
            print(extra_repr)
            parameters = np.sum([np.prod(p.size()) for p in self.parameters()])
            extra_repr = ', '.join((['parameters={}'.format(parameters)] if parameters > 0 else [])
                                 + ([extra_repr] if len(extra_repr) > 0 else []))
            # empty string will be split into list ['']
            if extra_repr:
                extra_lines = extra_repr.split('\n')
            child_lines = []
            for key, module in self._modules.items():
                mod_str = repr(module)
                mod_str = torch.nn.modules.module._addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
            lines = extra_lines + child_lines

            main_str = self._get_name() + '('
            if lines:
                # simple one-liner info, which most builtin Modules will use
                if len(extra_lines) == 1 and not child_lines:
                    main_str += extra_lines[0]
                else:
                    main_str += '\n  ' + '\n  '.join(lines) + '\n'

            main_str += ')'
            return main_str

        torch.nn.Module.__repr__ = __repr__

        return repr(module)
    else:
        counts = OrderedDict()
        params = OrderedDict()
        for m in module.modules():
            if m.__class__ not in [torch.nn.ModuleList, torch.nn.Sequential]:
                name = m.__class__.__name__
                counts[name] = counts.get(name, 0) + 1
                params[name] = params.get(name, 0) + np.sum([np.prod(p.size()) for p in m.parameters()]).astype(np.int32)
        title = '{:>3} {:<20} {}'.format('#', 'Module', 'Parameters')
        sep = '-' * 40
        return '\n'.join([title, sep] + ['{:>3} {:<20} {:,}'.format(counts[key], key, params[key]) for key in counts])