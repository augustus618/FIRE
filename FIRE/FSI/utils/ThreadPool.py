import queue
import threading
import time

from tqdm import tqdm

from .Tasks import AbstractTask


class ThreadPool:
    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self.task_queue: queue.Queue[AbstractTask] = queue.Queue()
        self.exit_event = threading.Event()
        self.threads = []

    def worker(self, worker_id: int):
        while not self.exit_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue
            print(f"Worker {worker_id} is processing task: {task.get_description()}", flush=True)
            task.run()
            self.task_queue.task_done()
            print(f"!!!Worker {worker_id} has done task: {task.get_description()}", flush=True)
            print(f'There are still {self.task_queue.qsize()} tasks left without threads to execute', flush=True)

    def start(self):
        for i in range(self.num_threads):
            thread = threading.Thread(target=self.worker, args=(i,))
            thread.start()
            self.threads.append(thread)

    def add_task(self, task: AbstractTask):
        self.task_queue.put(task)

    def wait_completion(self):
        self.task_queue.join()

    def stop(self):
        self.exit_event.set()

        for thread in self.threads:
            thread.join()

        print("All tasks are completed.")


class FixedTaskThreadPool:
    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self.task_queue: queue.Queue[AbstractTask] = queue.Queue()
        self.exit_event = threading.Event()
        self.threads = []
        self.progress_bar = None

    def worker(self, worker_id: int):
        while not self.exit_event.is_set():
            try:
                # time.sleep(random.randint(0, 3))
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                time.sleep(10)
                continue
            try:
                task.run()
            except Exception as e:
                print(f"{task.get_description()} meets an exception:{e}")
            self.task_queue.task_done()
            self.progress_bar.update(1)

    def start(self):
        self.progress_bar = tqdm(total=self.task_queue.qsize())
        for i in range(self.num_threads):
            thread = threading.Thread(target=self.worker, args=(i,))
            thread.start()
            self.threads.append(thread)

    def add_task(self, task: AbstractTask):
        self.task_queue.put(task)

    def wait_completion(self):
        self.task_queue.join()

    def stop(self):
        self.exit_event.set()

        for thread in self.threads:
            thread.join()

        print("All tasks are completed.")
