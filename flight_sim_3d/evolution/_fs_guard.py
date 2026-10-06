"""Imported only by the multiprocessing forkserver (via set_forkserver_preload): make the forkserver die with the
batch process. Workers set the same flag against the forkserver (sim.worker_init), so a SIGKILL of the batch
parent alone tears down the whole tree."""
try:
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
except Exception:  # noqa: BLE001
    pass
