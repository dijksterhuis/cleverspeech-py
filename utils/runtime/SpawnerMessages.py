import numpy as np
from cleverspeech.utils.Utils import log


def bytes_to_megabytes_str(x, power=2):
    return str(x // 1024 ** power) + " MB"


class SpawnerMessages(object):
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
