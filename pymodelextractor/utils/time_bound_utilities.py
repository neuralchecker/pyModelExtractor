import signal
from contextlib import contextmanager
from threading import Timer
import platform

def is_unix_system():
    curret_system = platform.system()
    return curret_system != 'Windows'

@contextmanager
def timeout(time):
    assert is_unix_system(), "timeout only supports UNIX systems for the moment as SIGALRM does not exist on Windows."
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        raise
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

#def timeout(time):
#    t = Timer(time, raise_timeout)
#    t.start()

def raise_timeout(signum, frame):
    raise TimeoutError