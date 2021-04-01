import numpy as np
from cleverspeech.utils.Utils import log, l_map


def bytes_to_megabytes_str(x, power=2):
    return str(x // 1024 ** power) + " MB"


class SpawnerMessages(object):
    def __init__(self):
        pass

    @staticmethod
    def reset():
        log("Restarted spawner variables.", wrap=True)

    @staticmethod
    def start_exiting():
        log("Exiting spawner.", wrap=False)

    @staticmethod
    def finishing_writes(queue_size):
        s = "Finishing up writes to disk.\nApprox. {} writes to go".format(
            queue_size
        )
        log(s, wrap=False)

    @staticmethod
    def finish_exiting():
        log("All data should now be written and processes closed.", wrap=True)

    @staticmethod
    def new_writer_process(pid):
        s = "Created new writer process with PID: {}".format(pid)
        log(s, wrap=True)

    @staticmethod
    def exit_writer_process(pid):
        s = "Exiting writer process with PID: {}".format(pid)
        log(s, wrap=True)

    @staticmethod
    def new_process(process):
        log("Spawned new attack with PID: {}".format(process.pid), wrap=True)

    @staticmethod
    def alive_process_count(n_spawn_attempts, n_processes_alive, max_processes):

        s = "Spawns attempt:\t{}\n".format(
            n_spawn_attempts
        )
        s += "Spawns alive:\t{}\n".format(
            n_processes_alive
        )
        s += "Spawns max:\t{}".format(
            max_processes
        )

        log(s, wrap=True)

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
        s = "Last attack process finished: "
        s += "Can spawn additional attacks now."
        log(s, wrap=False)

    @staticmethod
    def gpu_mem_usage_stats(total, used, free):

        s = "Overall GPU Memory:\n"
        s += "Total:\t{}\n".format(
            bytes_to_megabytes_str(
                total
            )
        )
        s += "Used:\t{}\n".format(
            bytes_to_megabytes_str(
                used
            )
        )
        s += "Free:\t{}".format(
            bytes_to_megabytes_str(
                free
            )
        )
        log(s, wrap=True)

    @staticmethod
    def gpu_process_usage_stats(compute_processes):

        s = "Per Process GPU Memory Usage:\n"

        def str_fmt(p, m):
            return "PID:\t{p}\tUSAGE: {m}".format(
                    p=p, m=bytes_to_megabytes_str(m)
                )

        s += "\n".join([str_fmt(p, m) for p, m in compute_processes.items()])

        log(s, wrap=True)

