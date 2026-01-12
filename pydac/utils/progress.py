"""Progress display utility for PyDAC"""


import sys

from typing import Optional

from contextlib import contextmanager


class ProgressBar:

    """Simple progress bar"""

    def __init__(self, total: int, desc: str = "Processing"):
        """
        Initialize progress bar

        Args:
            total: Total number of
            desc: Description
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self._last_percent = -1

    def update(self, n: int = 1):
        """Update progress"""

        self.current += n
        percent = int(100 * self.current / self.total)

        if percent != self._last_percent:
            self._last_percent = percent
            bar_length = 40
            filled = int(bar_length * self.current / self.total)
            bar = '=' * filled + '-' * (bar_length - filled)
            sys.stdout.write(f'\r{self.desc}: [{bar}] {percent}% ({self.current}/{self.total})')
            sys.stdout.flush()

    def finish(self):
        """Finish progress bar"""

        sys.stdout.write('\n')
        sys.stdout.flush()


@contextmanager
def progress_bar(total: int, desc: str = "Processing"):
    """Context manager for progress bar"""

    pb = ProgressBar(total, desc)
    try:
        yield pb
    finally:
        pb.finish()


class Spinner:
    """Simple spinner for indeterminate progress"""

    def __init__(self, desc: str = "Processing"):
        """
        Initialize spinner

        Args:
            desc: Description
        """
        self.desc = desc
        self.spinner_chars = "|/-\\"
        self.index = 0
        self.active = False

    def start(self):
        """Start spinner"""

        self.active = True
        self._spin()

    def stop(self):
        """Stop spinner"""

        self.active = False
        sys.stdout.write('\r' + ' ' * 60 + '\r')
        sys.stdout.flush()

    def _spin(self):
        """Spin animation"""

        if not self.active:
            return

        char = self.spinner_chars[self.index % len(self.spinner_chars)]
        sys.stdout.write(f'\r{self.desc}... {char}')
        sys.stdout.flush()
        self.index += 1

        import threading
        threading.Timer(0.1, self._spin).start()


@contextmanager
def spinner(desc: str = "Processing"):
    """Context manager for spinner"""

    sp = Spinner(desc)
    try:
        sp.start()
        yield sp
    finally:
        sp.stop()

