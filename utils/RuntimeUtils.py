import os
import time
from py3nvml import py3nvml as pynvml

import multiprocessing as mp
import tensorflow as tf
import numpy as np

from cleverspeech.data import Results
from cleverspeech.utils.Utils import log, run_decoding_check


class AttackFailedException(Exception):
    pass


class TFRuntime:
    def __init__(self, device_id=None):

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        tf.reset_default_graph()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True

        if not device_id:
            device = "/device:GPU:0"
        else:
            device = "/device:GPU:{}".format(device_id)

        self.session = tf.Session(config=self.config)
        self.device = tf.device(device)

    @staticmethod
    def log_attack_tensors():
        tensors = []
        for op in tf.get_default_graph().get_operations():
            if "qq" in op.name:
                for tensor in op.values():
                    tensors.append(tensor.__str__())
        return "\n".join(tensors)


def bytes_to_megabytes_str(x, power=2):
    return str(x // 1024 ** power) + " MB"


class GpuMemory:
    def __init__(self, gpu_device):

        pynvml.nvmlInit()
        self.__device_handle = pynvml.nvmlDeviceGetHandleByIndex(
            gpu_device
        )
        self.total = None
        self.used = None
        self.current_free = None
        self.previous_free = None
        self.max_batch = None
        self.all_batch = []

    def reset(self):
        self.used = None
        self.current_free = None
        self.previous_free = None
        self.max_batch = None
        self.all_batch = []

    def update_usage(self):

        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(
            self.__device_handle
        )

        self.total = gpu_memory.total
        self.used = gpu_memory.used

        if self.current_free is not None:
            self.previous_free = self.current_free

        self.current_free = gpu_memory.free

    def update_batch(self):
        self.all_batch.append(self.previous_free - self.current_free)
        self.max_batch = max(self.all_batch)

    def check_resource(self):
        return self.max_batch > self.current_free * 0.9


class Processes:
    def __init__(self, max_processes):

        self.total_cpu = mp.cpu_count()
        self.processes = None
        self.alive = None
        self.attempts = None

        if max_processes is not None:
            assert type(max_processes) is int
            self.max = max_processes
        else:
            self.max = self.total_cpu // 2

        self.reset()

    def reset(self):
        self.processes = []
        self.alive = 0
        self.attempts = 0

    def new(self, results_queue, args):

        parent_conn, worker_conn = mp.Pipe()
        args = (results_queue, worker_conn, *args)
        p = mp.Process(
            target=boilerplate,
            args=args,
        )
        p.start()
        self.processes.append((p, parent_conn))
        self.attempts += 1
        return p

    def block(self):

        # block until all current processes have completed
        for process, _ in self.processes:
            process.join()

    def terminate(self):
        for process, _ in self.processes:
            process.terminate()

    def check_last(self):
        _, pipe = self.processes[-1]
        return pipe.recv()

    def check_alive(self):
        self.alive = 0
        for process, _ in self.processes:
            if process.is_alive():
                self.alive += 1

    def check_resource(self):
        return self.attempts >= self.max


class AttackSpawnerMessages(object):
    def __init__(self):
        pass

    @staticmethod
    def reset():
        log("Restarted spawner variables.", wrap=False)

    @staticmethod
    def new_process(process):
        log("Spawned new attack with PID: {}".format(process.pid), wrap=False)

    @staticmethod
    def alive_process_count(n_spawn_attempts, n_processes_alive, max_processes):

        s = "Current spawn attempts: {}\n".format(
            n_spawn_attempts
        )
        s += "Current alive processes: {}\n".format(
            n_processes_alive
        )
        s += "Max spawns: {}".format(
            max_processes
        )

        log(s, wrap=False)

    @staticmethod
    def healthy(p):
        s = "Last attack with PID {} is healthy.".format(p.pid)
        log(s, wrap=False)

    @staticmethod
    def unhealthy(p):
        s = "Last attack with PID {} is unhealthy!".format(p.pid)
        log(s, wrap=False)

    @staticmethod
    def waiting(delay):
        s = "Waiting {n} seconds for warm up before spawning again.".format(
            n=delay
        )
        log(s, wrap=True)

    @staticmethod
    def start_blocking():
        s = "No more resource available: "
        s += "Blocking until current attacks completed."
        log(s, wrap=False)

    @staticmethod
    def stop_blocking():
        s = "\nLast attack process finished: "
        s += "Will spawn additional attacks now."
        log(s, wrap=False)

    @staticmethod
    def gpu_mem_usage_stats(free, previous, batches):

        mem_mean = np.mean(batches)
        mem_max = np.max(batches)
        mem_min = np.min(batches)

        mem_all = "\t".join([bytes_to_megabytes_str(x) for x in batches])

        s = "GPU Memory Usage:\n"

        s += "Free:\t{}\n".format(
            bytes_to_megabytes_str(
                free
            )
        )
        s += "Last:\t{}\n".format(
            bytes_to_megabytes_str(
                previous
            )
        )
        s += "Max:\t{}\n".format(
            bytes_to_megabytes_str(
                mem_max
            )
        )
        s += "Min:\t{}\n".format(
            bytes_to_megabytes_str(
                mem_min
            )
        )
        s += "Mean:\t{}\n".format(
            bytes_to_megabytes_str(
                mem_mean
            )
        )

        s += "All:\t" + mem_all
        log(s, wrap=True)


class AttackSpawner:
    def __init__(self, gpu_device=0, max_processes=None, delay=120, file_writer=None):

        self.delay = delay
        self.device = gpu_device

        self.processes = Processes(max_processes)
        self.gpu_memory = GpuMemory(self.device)
        self.__results_queue = mp.Queue()
        if file_writer is not None:
            self.__writer_process = mp.Process(
                target=file_writer.write, args=(self.__results_queue,)
            )
            self.__writer_process.start()
        else:
            self.__writer_process = None

        self.__messenger = AttackSpawnerMessages()

        self.__reset__()

    def __reset__(self):
        self.processes.reset()
        self.gpu_memory.reset()
        self.__messenger.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.processes.terminate()
            self.__results_queue.close()
            if self.__writer_process is not None:
                self.__writer_process.terminate()
            raise AttackFailedException(
                "Attack Failed:\n\n{v}\n{t}".format(v=exc_val, t=exc_tb)
            )
        else:
            self.__results_queue.close()
            if self.__writer_process is not None:
                self.__writer_process.close()

    def __wait(self):
        self.__messenger.waiting(self.delay)
        time.sleep(self.delay)

    def spawn(self, *args):

        self.gpu_memory.update_usage()

        p = self.processes.new(self.__results_queue, args)

        self.__messenger.new_process(p)

        child_status = self.processes.check_last()

        if child_status is False:
            self.__messenger.unhealthy(p)
        else:
            self.__messenger.healthy(p)

        self.__wait()
        self.processes.check_alive()
        self.gpu_memory.update_usage()
        self.gpu_memory.update_batch()

        self.__messenger.alive_process_count(
            self.processes.attempts,
            self.processes.alive,
            self.processes.max
        )

        self.__messenger.gpu_mem_usage_stats(
            self.gpu_memory.current_free,
            self.gpu_memory.max_batch,
            self.gpu_memory.all_batch,
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


def boilerplate(results_queue, healthy_conn, settings, attack_fn, batch):

    # we *must* call the tensorflow session within the batch loop so the
    # graph gets reset: the maximum example length in a batch affects the
    # size of most graph elements.

    # tensorflow sessions can't be passed between processes either, so we have
    # to create it here.

    try:
        tf_runtime = TFRuntime(settings["gpu_device"])
        with tf_runtime.session as sess, tf_runtime.device as tf_device:

            # Initialise curried attack graph constructor function

            attack = attack_fn(sess, batch, settings)

            # log some useful things for debugging before the attack runs

            run_decoding_check(attack, batch)

            log(
                "Created Attack Graph and Feeds. Loaded TF Operations:",
                wrap=False
            )
            log(funcs=tf_runtime.log_attack_tensors)

            # Run the attack generator loop. See `Attacks/Procedures.py` for
            # detailed info on returned results.
            log(
                "Beginning attack run...\nMonitor progress in: {}".format(
                    settings["outdir"] + "log.txt"
                )
            )

            # Inform the parent process that we've successfully loaded the graph
            # and will start the attacks.
            healthy_conn.send(True)
            attack.run(results_queue)

    except tf.errors.ResourceExhaustedError as e:

        # Fail gracefully for OOM GPU issues.

        s = "Out of GPU Memory! Attack failed to run for these examples:\n"
        s += '\n'.join(batch.audios["basenames"])
        s += "\n\nError Traceback:\n{e}".format(e=e)

        log(s, wrap=True)
        healthy_conn.send(False)
        raise

    except Exception as e:

        # We shouldn't use a broad Exception, but OOM errors are the most common
        # point of breakage right now.

        # if there's a non-OOM exception then something broke with the code that
        # needs to be fixed

        s = "Something broke! Attack failed to run for these examples:\n"
        s += '\n'.join(batch.audios["basenames"])
        s += "\n\nError Traceback:\n{e}".format(e=e)
        s += "\n\nExiting Hard!"

        log(s, wrap=True)
        healthy_conn.send(False)

        exit(100)

