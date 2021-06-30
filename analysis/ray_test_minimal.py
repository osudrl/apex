import numpy as np
import ray
import time
import multiprocessing as mp

def reg_sleep(length):
    time.sleep(length)

@ray.remote
def ray_sleep(length):
    time.sleep(length)

num_proc = 50
length = 0.5
seq_start = time.time()
for i in range(num_proc):
    reg_sleep(length)
print("Regular total time: ", time.time() - seq_start)

pool = mp.Pool(num_proc)
args = [length]*num_proc
mp_start = time.time()
pool.map(reg_sleep, args)
print("MP total time: ", time.time() - mp_start)
pool = None

ray.init(num_cpus=num_proc)
ray_start = time.time()
ids = [ray_sleep.remote(length) for i in range(num_proc)]
ray.get(ids)
print("ray total time: ", time.time() - ray_start)