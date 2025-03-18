import time
from logging import Logger
import os
import functools


def timeit(logger: Logger = None, print_time: bool = False, return_val: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            file_name = os.path.basename(func.__code__.co_filename)
            message = f"{func.__name__} (file: {file_name}) executed in {elapsed_time:.4f} seconds"
            
            if logger:
                logger.info(message)
            if print_time:
                print(message)
                
            if return_val:
                return result, elapsed_time
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    @timeit(print_time=True, return_val=True)
    def sample_function(n):
        total = 0
        for i in range(n):
            total += i
        return total

    res, exec_time = sample_function(1000000)
    print(f"{res=}")
    print(f"{exec_time=}")