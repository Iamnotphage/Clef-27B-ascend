"""First-call-only host trace: unbuffered per-rank append, no device imports."""
import json
import os
from pathlib import Path
import time

FD_ENV = 'CLEF_GROUPED29_TRACE_FD'


class FirstCallTrace:
    def __init__(self, path, rank, enabled=True):
        self.path, self.rank, self.fd = Path(path), rank, None
        self.previous_fd_env = None
        if not enabled:
            return
        if rank not in (0, 1) or not self.path.is_absolute():
            raise ValueError('First trace requires an absolute rank0/1 file')
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                          os.O_APPEND | os.O_CLOEXEC, 0o640)
        self.previous_fd_env = os.environ.get(FD_ENV)
        os.environ[FD_ENV] = str(self.fd)

    def emit(self, stage, **fields):
        if self.fd is None:
            return
        record = {'schema': 'grouped29-firstcall/1', 'source': 'python',
                  'rank': self.rank, 'pid': os.getpid(), 'pgid': os.getpgrp(),
                  'monotonic_ns': time.monotonic_ns(), 'stage': stage, **fields}
        data = (json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if len(data) > 4096:
            raise ValueError('Trace record exceeds atomic append budget')
        # No Python/stdio buffer. A complete line reaches the kernel here.
        if os.write(self.fd, data) != len(data):
            raise OSError('Short first-call trace write')

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)
            if self.previous_fd_env is None:
                os.environ.pop(FD_ENV, None)
            else:
                os.environ[FD_ENV] = self.previous_fd_env
