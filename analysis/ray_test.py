import numpy as np
import ray
import time
import threading

@ray.remote
class test_worker(object):
    def __init__(self, id_num):
        self.id_num = id_num
        self._thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def finish(self):
        id_num = self._thread.join()
        return id_num

    def run(self):
        print("Started worker ", self.id_num)
        time.sleep(10)
        print("finished inside run")
        return self.id_num


def reg_sum(size):
    start_t = time.time()
    sum = 0
    for i in range(size):
        sum += 1
    print("reg run time: ", time.time() - start_t)
    return sum

@ray.remote
def ray_sum(size):
    start_t = time.time()
    sum = 0
    for i in range(size):
        sum += 1
    print("ray run time: ", time.time() - start_t)
    return sum


ray.init(num_cpus=4)
# workers = [test_worker.remote(i) for i in range(4)]
worker = test_worker.remote(0)
print("worker", worker)
# worker.start.remote()
result_id = worker.run.remote()
print("result id:", result_id)
# time.sleep(2)

# result_id = worker.run.remote()
time.sleep(2)
# ray.kill(worker)
# worker.kill()
# result_id.kill()
ray.cancel(result_id)
print("killed worker")
result_id = worker.run.remote()
# time.sleep(2)
ray.get(result_id)
print("done")
# ray.kill(worker)
# ray.get(result_id)
# print("got result id")

# num_proc = 4
# seq_start = time.time()
# for i in range(num_proc):
#     reg_sum(100000)
# print("Regular total time: ", time.time() - seq_start)


# ray.init(num_cpus=num_proc)
# ray_start = time.time()
# ids = [ray_sum.remote(100000) for i in range(num_proc)]
# ray.get(ids)
# print("ray total time: ", time.time() - ray_start)