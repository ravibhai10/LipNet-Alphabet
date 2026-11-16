"""Process measurements with the standard library; no inferred benchmark values."""
import ctypes
import os
import time


def memory_bytes():
    if os.name=='nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
        api=ctypes.WinDLL('psapi');api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        if api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            return {'rss_bytes':counters.WorkingSetSize,'peak_rss_bytes':counters.PeakWorkingSetSize}
    return {'rss_bytes':None,'peak_rss_bytes':None}


class Measurement:
    def __init__(self):self.wall=time.perf_counter();self.cpu=time.process_time();self.before=memory_bytes()
    def finish(self):
        wall=time.perf_counter()-self.wall;cpu=time.process_time()-self.cpu
        return {'wall_ms':round(wall*1000,2),'process_cpu_ms':round(cpu*1000,2),
                'process_cpu_percent_one_core':round(cpu/max(wall,1e-9)*100,2),
                'requests_per_second_sequential':round(1/max(wall,1e-9),3),
                'memory_before':self.before,'memory_after':memory_bytes(),
                'measurement_scope':'Current process; one request, not a sustained-load benchmark.'}
