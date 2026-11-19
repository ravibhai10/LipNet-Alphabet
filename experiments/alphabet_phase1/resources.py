"""Small Windows resource probes without installing psutil."""
import ctypes
import shutil
from pathlib import Path


def snapshot(path=Path.cwd()):
    class Memory(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                    ('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),
                    ('page',ctypes.c_ulonglong),('apage',ctypes.c_ulonglong),
                    ('virtual',ctypes.c_ulonglong),('avirtual',ctypes.c_ulonglong),('extra',ctypes.c_ulonglong)]
    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('Native memory probe failed')
    return {'available_ram_bytes': value.available, 'total_ram_bytes': value.total,
            'memory_load_percent': value.load, 'free_disk_bytes': shutil.disk_usage(path).free}


def guard(minimum_ram_mib=512):
    state = snapshot()
    if state['free_disk_bytes'] < 20 * 1024**3:
        raise RuntimeError('STOP: free disk below 20 GiB')
    if state['available_ram_bytes'] < minimum_ram_mib * 1024**2:
        raise RuntimeError(f'STOP: available RAM below {minimum_ram_mib} MiB for this bounded stage: {state}')
    return state


def process_memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)] + [
            (name,ctypes.c_size_t) for name in ['peak_rss','rss','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
    state = Counters()
    state.cb = ctypes.sizeof(state)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle),ctypes.byref(state),state.cb):
        raise OSError('Process memory probe failed')
    return {'rss_bytes':state.rss,'peak_rss_bytes':state.peak_rss,'pagefile_bytes':state.pagefile}
