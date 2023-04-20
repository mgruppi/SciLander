import time


def timed(func):
    """
    Decorator to time stuff.
    :param func: function to be called
    :return:
    """
    def timed_func(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print(" - Function %s - %.2f seconds" % (func.__name__, t1-t0))
        return result
    return timed_func
