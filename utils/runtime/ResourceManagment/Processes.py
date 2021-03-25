import traceback

import multiprocessing as mp
import tensorflow as tf

from cleverspeech.utils.Utils import log, run_decoding_check
from cleverspeech.utils.runtime.TensorflowRuntime import TFRuntime


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

    def close(self):
        for process, _ in self.processes:
            if process.is_alive():
                process.close()

    def terminate(self):
        for process, _ in self.processes:
            if process.is_alive():
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


def boilerplate(results_queue, healthy_conn, settings, attack_fn, batch):

    # we *must* call the tensorflow session within the batch loop so the
    # graph gets reset: the maximum example length in a batch affects the
    # size of most graph elements.

    # tensorflow sessions can't be passed between processes either, so we have
    # to create it here (at least, not easily).

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

            log(
                "Beginning attack run...\nMonitor progress in: {}".format(
                    settings["outdir"] + "log.txt"
                )
            )

            # Start the attacks.
            attack.run(results_queue, healthy_conn)

    except tf.errors.ResourceExhaustedError as e:

        # Fail gracefully for OOM GPU issues.
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))

        s = "Out of GPU Memory! Attack failed to run for these examples:\n"
        s += '\n'.join(batch.audios["basenames"])
        s += "\n\nError Traceback:\n{e}".format(e=tb)

        log(s, wrap=True)
        healthy_conn.send(True)

    except Exception as e:

        # We shouldn't use a broad Exception, but OOM errors are the most common
        # point of breakage right now.

        # if there's a non-OOM exception then something broke with the code that
        # needs to be fixed

        tb = "".join(traceback.format_exception(None, e, e.__traceback__))

        s = "Something broke! Attack failed to run for these examples:\n"
        s += '\n'.join(batch.audios["basenames"])
        s += "\n\nError Traceback:\n{e}".format(e=tb)

        log(s, wrap=True)
        healthy_conn.send(False)

