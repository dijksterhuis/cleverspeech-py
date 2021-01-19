import time
from py3nvml import py3nvml as pynvml

import multiprocessing as mp
import tensorflow as tf

from cleverspeech.utils.Utils import log


def create_tf_runtime(device_id=None):

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    if not device_id:
        device = "/device:GPU:0"
    else:
        device = "/device:GPU:{}".format(device_id)

    return tf.Session(config=config), tf.device(device)


def log_attack_tensors():
    tensors = []
    for op in tf.get_default_graph().get_operations():
        if "qq" in op.name:
            for tensor in op.values():
                tensors.append(tensor.__str__())
    return "\n".join(tensors)


class AttackSpawner:
    def __init__(self, gpu_device=0, max_processes=None, delay=120):

        self.gpu_device = gpu_device
        self.max_processes = max_processes
        self.delay = delay

        if max_processes is not None:
            assert type(max_processes) is int
            self.max_processes = max_processes
        else:
            self.max_processes = mp.cpu_count() // 2

        self.__current_gpu_memory_free = None
        self.__previous_gpu_memory_free = None
        self.__previous_batch_gpu_memory = None

        self.__total_cpus = mp.cpu_count()
        self.__processes = []

        pynvml.nvmlInit()

        self.__device_handle = pynvml.nvmlDeviceGetHandleByIndex(
            self.gpu_device
        )

    def __get_current_gpu_memory(self):

        if self.__current_gpu_memory_free is not None:
            self.__previous_gpu_memory_free = self.__current_gpu_memory_free

        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(
            self.__device_handle
        )

        self.__current_gpu_memory_free = gpu_memory.free

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO
        pass

    def __create_process(self, attack_run, args):

        # we *must* call the tensorflow session within the batch loop so the
        # graph gets reset: the maximum example length in a batch affects the
        # size of most graph elements.
        try:
            p = mp.Process(
                target=attack_run,
                args=args
            )
            p.start()
            log(
                "\nSpawned new attack with PID: {}".format(
                    p.pid
                ),
                wrap=False
            )

            # wait for a while to make sure the graph is fully loaded onto the GPU
            self.__processes.append(p)

        except tf.errors.ResourceExhaustedError as e:

            log("Out of Memory! Blocking until current processes complete!")
            log("Error Traceback:\n{e}".format(e=e))
            self.__block_until_completed()
            self.__create_process(attack_run, args)

    def __wait(self):
        log(
            "Waiting {} seconds before checking for free resources.".format(
                self.delay
            ),
            wrap=True
        )
        time.sleep(self.delay)

    def __get_last_batch_gpu_memory(self):
        previous = self.__previous_gpu_memory_free
        current = self.__current_gpu_memory_free
        self.__previous_batch_gpu_memory = previous - current

    def __check_gpu_available(self):
        current = self.__current_gpu_memory_free
        batch = self.__previous_batch_gpu_memory
        return batch > current

    def __check_cpu_available(self):
        return len(self.__processes) >= self.max_processes

    def __block_until_completed(self):

        # block until all current processes have completed
        for process in self.__processes:
            process.join()

        s = "\nLast attack process finished: "
        s += "Will spawn additional attacks now."

        log(s)

        # once finished, clear processes and start again
        self.__processes.clear()

    def spawn(self, attack_run, *args):

        self.__get_current_gpu_memory()
        self.__create_process(attack_run, args)
        self.__wait()
        self.__get_current_gpu_memory()
        self.__get_last_batch_gpu_memory()

        if self.__check_gpu_available() or self.__check_cpu_available():

            s = "\nNo more resource available: "
            s += "Blocking until current attacks completed.\n"

            s += "Current N spawned processes: {}\t".format(
                len(self.__processes)
            )
            s += "Max N spawned processes: {}\n".format(
                self.max_processes
            )
            s += "Current free gpu memory: {}\t".format(
                self.__current_gpu_memory_free
            )
            s += "Last attack used gpu memory: {}".format(
                self.__previous_batch_gpu_memory
            )

            log(s)

            self.__block_until_completed()

        else:
            # Not full so don't block yet.
            pass
