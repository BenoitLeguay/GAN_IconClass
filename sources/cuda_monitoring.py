import torch
import gc
import sys
import inspect
import pynvml
import math


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def get_tensor_info():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), "\t", obj.device, "\t", convert_size(sys.getsizeof(obj.storage())), tuple(obj.size()), "\t")
        except:
            pass


def get_gpu_memory_info():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {convert_size(info.total)}')
    print(f'free     : {convert_size(info.free)}')
    print(f'used     : {convert_size(info.used)}\t({round((info.used/info.free*100), 2)}%)')