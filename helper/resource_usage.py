import os
import sys


def process_memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        size = ctypes.c_size_t

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", size),
                ("WorkingSetSize", size),
                ("QuotaPeakPagedPoolUsage", size),
                ("QuotaPagedPoolUsage", size),
                ("QuotaPeakNonPagedPoolUsage", size),
                ("QuotaNonPagedPoolUsage", size),
                ("PagefileUsage", size),
                ("PeakPagefileUsage", size),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if read(current(), ctypes.byref(counters), counters.cb):
            return {
                "process_rss_bytes": int(counters.WorkingSetSize),
                "process_peak_rss_bytes": int(counters.PeakWorkingSetSize),
                "memory_scope": "shared_process_snapshot_and_lifetime_peak",
            }
        return {"memory_scope": "unavailable"}
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "process_peak_rss_bytes": int(peak * (1 if sys.platform == "darwin" else 1024)),
            "memory_scope": "shared_process_lifetime_peak",
        }
    except (ImportError, OSError):
        return {"memory_scope": "unavailable"}
