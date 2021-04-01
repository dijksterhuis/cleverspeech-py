import time
import multiprocessing as mp

from cleverspeech.utils.runtime.ResourceManagment.Processes import Processes
from cleverspeech.utils.runtime.ResourceManagment.GPU import GpuMemory
from cleverspeech.utils.runtime.SpawnerMessages import SpawnerMessages


class AttackFailedException(Exception):
    pass


class AttackSpawner:
    def __init__(self, gpu_device=0, max_processes=None, delay=120, file_writer=None):

        self.delay = delay
        self.device = gpu_device

        self.processes = Processes(max_processes)
        self.gpu_memory = GpuMemory(self.device)

        self.__results_queue = mp.Queue()
        self.__messenger = SpawnerMessages()

        if file_writer is not None:
            self.__writer_process = mp.Process(
                target=file_writer.write, args=(self.__results_queue,)
            )
            self.__writer_process.start()
            self.__messenger.new_writer_process(self.__writer_process.pid)
        else:
            self.__writer_process = None

        self.__reset__()

    def __reset__(self):
        self.processes.reset()
        self.gpu_memory.reset()
        self.__messenger.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        # hit everything with a hammer to kill processes.
        self.processes.terminate()
        self.__results_queue.put(False)
        self.__writer_process.join()
        self.__results_queue.close()
        self.__writer_process.terminate()
            
    def __wait(self):
        self.__messenger.waiting(self.delay)
        time.sleep(self.delay)

    def spawn(self, *args):

        self.gpu_memory.update_usage()

        p = self.processes.new(self.__results_queue, args)

        self.__messenger.new_process(p)

        child_status = self.processes.check_last()

        if child_status is False:
            # Something broke, cause an exception in main process.
            self.__messenger.unhealthy(p)
            raise AttackFailedException(
                "Child Process {} was unhealthy. Unsafe to continue".format(p.pid)
            )
        else:
            self.__messenger.healthy(p)

        self.__wait()

        self.processes.check_alive()

        self.gpu_memory.update_usage()
        self.gpu_memory.update_pid_usage(self.processes.processes)
        self.gpu_memory.update_process_stats()

        self.__messenger.alive_process_count(
            self.processes.attempts,
            self.processes.alive,
            self.processes.max
        )

        self.__messenger.gpu_mem_usage_stats(
            self.gpu_memory.total,
            self.gpu_memory.used,
            self.gpu_memory.free,
        )

        self.__messenger.gpu_process_usage_stats(
            self.gpu_memory.pid_mem_usage,
        )

        gpu = self.gpu_memory.check_resource()
        cpu = self.processes.check_resource()
        fail = not child_status

        if gpu or cpu or fail:
            # once finished, reset all the __restart__ variables and start again
            self.__messenger.start_blocking()
            self.processes.block()
            self.__reset__()
            self.__messenger.stop_blocking()

        else:
            # Not full so don't block yet.
            pass

