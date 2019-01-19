import colorama
from collections import defaultdict, OrderedDict
import torch
import numpy as np
import builtins

is_profile = 'profile' in builtins.__dict__
profile = builtins.__dict__.get('profile', lambda x: x)

# def profile():
#     def wrapper(f):
#         @wraps(f)
#         def inner(*args, **kwargs):
#             profile = builtins.__dict__.get('profile', lambda x: x)(fields, envelope)
#             profile(f)(*args, **kwargs)
#             return 
#         return inner
#     return wrapper

def logv(string=""):
    import inspect
    frameinfo = inspect.getframeinfo(inspect.currentframe().f_back)
    memusage = torch.cuda.memory_allocated()
    maxmemusage = torch.cuda.max_memory_allocated()
    print('[{}/{}]:{} {:.3f}gb gpu memusage {}gb max {}'.format(frameinfo.filename, frameinfo.function, frameinfo.lineno, memusage * 1e-9,  maxmemusage * 1e-9, string))

def block2mat(Mblock):
    Nr,Nc,bh,bw = Mblock.shape
    M = np.zeros((Nr*bh,Nc*bw))
    for k in range(Nr):
        M[k*bh:(k+1)*bh] = np.hstack(Mblock[k])
        
    return M

def blockTranspose(M,blockHeight,blockWidth):
    """
    Switches block indices without transposing the blocks
    """
    r,c = M.shape
    Nr = int(r / blockHeight)
    Nc = int(c / blockWidth)
    Mblock = np.zeros((Nr,Nc,blockHeight,blockWidth))
    for i in range(Nr):
        for j in range(Nc):
            Mblock[i,j] = M[i*blockHeight:(i+1)*blockHeight,j*blockWidth:(j+1)*blockWidth]
            
            
    MtBlock = np.zeros((Nc,Nr,blockHeight,blockWidth))
    for i in range(Nr):
        for j in range(Nc):
            MtBlock[j,i] = Mblock[i,j]
            
    return block2mat(MtBlock)

def blockHankel(Hleft,Hbot=None,blockHeight=1):
    """
    Compute a block hankel matrix from the left block matrix and the optional bottom block matrix
    
    Hleft is a matrix of dimensions (NumBlockRows*blockHeight) x blockWidth
    
    Hbot is a matrix of dimensions blockHeight x (NumBlockColumns*blockWidth)
    """
    
    blockWidth = Hleft.shape[1]
    if Hbot is None:
        Nr = int(len(Hleft) / blockHeight)
        Nc = Nr
    else:
        blockHeight = len(Hbot)
        Nr = int(len(Hleft) / blockHeight)
        Nc = int(Hbot.shape[1] / blockWidth)
        
    LeftBlock = np.zeros((Nr,blockHeight,blockWidth))
    
    for k in range(Nr):
        LeftBlock[k] = Hleft[k*blockHeight:(k+1)*blockHeight]
        
    
        
    # Compute hankel matrix in block form
    MBlock = np.zeros((Nr,Nc,blockHeight,blockWidth))
    
    for k in range(np.min([Nc,Nr])):
        # If there is a bottom block, could have Nc > Nr or Nr > Nc
        MBlock[:Nr-k,k] = LeftBlock[k:]
        
        
    if Hbot is not None:
        BotBlock = np.zeros((Nc,blockHeight,blockWidth))
        for k in range(Nc):
            BotBlock[k] = Hbot[:,k*blockWidth:(k+1)*blockWidth]
            
        for k in range(np.max([1,Nc-Nr]),Nc):
            MBlock[Nr-Nc+k,Nc-k:] = BotBlock[1:k+1]
            
    
    # Convert to a standard matrix
    M = block2mat(MBlock)
        
    return M

def getHankelMatrices(x,NumRows,NumCols,blockWidth=1):
    # For consistency with conventions in Van Overschee and De Moor 1996, 
    # it is assumed that the signal at each time instant is a column vector
    # and the number of samples is the number of columns.
    
    bh = len(x)
    bw = 1
    xPastLeft = blockTranspose(x[:,:NumRows],blockHeight=bh,blockWidth=bw)
    XPast = blockHankel(xPastLeft,x[:,NumRows-1:NumRows-1+NumCols])
    
    xFutureLeft = blockTranspose(x[:,NumRows:2*NumRows],blockHeight=bh,blockWidth=bw)
    XFuture = blockHankel(xFutureLeft,x[:,2*NumRows-1:2*NumRows-1+NumCols])
    return XPast,XFuture

def N4SID_simple(y,NumRows,NumCols,NSig):
    NumOutputs = y.shape[0]
    YPast,YFuture = getHankelMatrices(y,NumRows,NumCols)    
    
    L = np.linalg.lstsq(YPast.T,YFuture.T, rcond=1e-4)[0].T
    Z = L @ YPast
    LShift = np.linalg.lstsq(YPast.T,YFuture[NumOutputs:].T, rcond=1e-4)[0].T
    
    ZShift = LShift @ YPast
    
    U, S, Vt = np.linalg.svd(np.dot(L,YPast), full_matrices=False)

    Gamma = U[:, :NSig] * np.sqrt(S[None, :NSig])
    GammaLess = Gamma[:-NumOutputs]
    GamShiftSolve = np.linalg.lstsq(GammaLess,ZShift, rcond=1e-4)[0]
    GamSolve = np.linalg.lstsq(Gamma,Z, rcond=1e-4)[0]

    GamYData = np.vstack((GamShiftSolve,YFuture[:NumOutputs]))

    K = np.linalg.lstsq(GamSolve.T,GamYData.T, rcond=1e-4)[0].T
    
    AID = K[:NSig,:NSig]
    CID = K[NSig:,:NSig]
    
    return AID, CID

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