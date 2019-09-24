import ray
import time

@ray.remote
class Actor():
    def __init__(self, memory_id, learner_id):
        self.counter = 0
        self.memory_id = memory_id
        self.learner_id = learner_id

    def collect_experience(self):
        bar = False
        while True:
            #print("collecting")
            # simulate load of stepping through environment
            time.sleep(0.01)
            # increment counter
            self.counter += 1
            # send new data to memory server
            self.memory_id.receive_experience.remote(self.counter)

            # periodically query learner
            if self.counter % 5 == 0:
                foo, bar = ray.get(self.learner_id.send_step_count.remote())
                print("got stuff from learner")

            if bar:
                print("quitting collection loop")
                return

@ray.remote
class MemoryServer():
    def __init__(self):
        self.counter = 0

    def receive_experience(self, data):
        self.counter += data

    def send_experience(self):
        return self.counter

@ray.remote
class Evaluator():
    def __init__(self):
        self.counter = 0

    def evaluate_loop(self, learner_id):
        bar = False
        while True:
            time.sleep(0.5)
            foo, bar = ray.get(learner_id.send_step_count.remote())
            print("Evaluating: {}".format(foo))
            if bar:
                return

@ray.remote
class Learner():
    def __init__(self, memory_id, eval_id):
        self.steps = 0
        self.done = False
        self.num_train_steps = 100
        self.memory_id = memory_id
        self.eval_id = eval_id

    def train(self):
        while self.steps < self.num_train_steps:
            # collect experience from memory
            memory_data = ray.get(self.memory_id.send_experience.remote())
            # simulate optimization
            time.sleep(0.25)
            # increment number of steps
            self.steps += 1

        print("exiting learner")
        self.done = True

    def send_step_count(self):
        return self.steps, self.done


if __name__ == "__main__":
    ray.init()
    
    # create many Actors, one Memory Server, one Evaluator, one Learner
    memory_id = MemoryServer.remote()
    eval_id = Evaluator.remote()
    learner_id = Learner.remote(memory_id, eval_id)
    actor_ids = [Actor.remote(memory_id, learner_id) for _ in range(10)]

    # Start actor collection loop and learner train loop and evaluator loop
    futures = [actor_id.collect_experience.remote() for actor_id in actor_ids]
    futures.append(learner_id.train.remote())
    futures.append(eval_id.evaluate_loop.remote(learner_id))

    ray.wait(futures, num_returns=len(futures))

    print("All done.")

