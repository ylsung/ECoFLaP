from time import time

def print_time(func):
    def wrapper(*args, **kwargs):
        start = time()

        ret = func(*args, **kwargs)
        
        time_spent = time() - start
        
        print(f"{func.__name__} spent {time_spent:.3f} s")
        
        return ret
    
    return wrapper