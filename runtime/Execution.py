import os
import traceback
import multiprocessing as mp

from progressbar import ProgressBar

from cleverspeech.data.egress import load
from cleverspeech.runtime.TensorflowRuntime import TFRuntime
from cleverspeech.utils.Utils import log, Logger


RESULTS_WRITER_FUNCS = {
    "local_latest": load.write_latest_metadata_to_local_json_file,
    "local_all": load.write_all_metadata_to_local_json_files,
    "s3_latest": load.write_latest_metadata_to_s3,
    "s3_all": load.write_all_metadata_to_s3
}

SETTINGS_WRITER_FUNCS = {
    "local": load.write_settings_to_local_json_file,
    "s3": load.write_settings_to_s3,
}


class EndOfRunException(Exception):
    pass


def _check_for_dead_queue(results):
    return results == "dead"


def _results_generator(results_writer, results, settings):
    from cleverspeech.data.egress.transform import transforms_gen
    for example in transforms_gen(results, settings):
        if example["success"] is True:
            results_writer(settings["outdir"], example)


def writer_boilerplate_fn(queue, settings):

    import traceback
    from time import sleep

    try:

        settings_writer_fn = SETTINGS_WRITER_FUNCS[settings["writer"].split("_")[0]]
        settings_writer_fn(settings["outdir"], settings)

        results_writer_fn = RESULTS_WRITER_FUNCS[settings["writer"]]

        while True:

            sleep(1)
            results = queue.get()

            if _check_for_dead_queue(results):
                queue.task_done()
                raise EndOfRunException

            else:
                _results_generator(results_writer_fn, results, settings)
                queue.task_done()

    except EndOfRunException:
        Logger.info(
            "Dead queue entry detected... Writer subprocess exiting."
        )

    except KeyboardInterrupt:
        Logger.critical(
            "KeyboardInterrupt detected... Writer subprocess exiting."
        )

    except BaseException as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        Logger.critical("Something broke during file writes!")
        Logger.critical("{e}".format(e=tb))
        raise e


def executor_boilerplate_fn(results_queue, settings, batch, attack_fn):

    from cleverspeech.data.egress.extract import get_attack_state

    # tensorflow sessions can't be passed between processes
    tf_runtime = TFRuntime(settings)

    with tf_runtime.session as sess, tf_runtime.device as tf_device:

        # Initialise attack graph constructor function
        attack = attack_fn(sess, batch, settings)

        # log some useful things for debugging before the attack runs
        attack.validate()

        s = "Created Attack Graph and Feeds. Loaded TF Operations:"
        Logger.info(s, timings=True)
        Logger.log(
            funcs=tf_runtime.log_attack_tensors, prefix="\n", postfix="\n"
        )

        s = "Beginning attack run... Monitor progress in: {}".format(
            os.path.join(settings["outdir"], "log.txt")
        )
        Logger.info(s, timings=True)

        if settings["dry_run"] is True:
            return None

        try:

            with ProgressBar(min_value=0, max_value=1+settings["nsteps"]) as p:

                for step, is_results_step, successes in attack.run():

                    if step > 0: p.update(step)

                    if is_results_step and settings["no_step_logs"] is False:
                        res = get_attack_state(attack, successes)
                        results_queue.put(res)

        except BaseException:
            s = "Attack failed to run for these examples: "
            s += ', '.join(batch.audios["basenames"])
            Logger.critical(s, timings=True)
            raise


def manager(settings, attack_fn, batch_gen):

    # modify this as late as possible to catch any added directories in exp defs
    settings["outdir"] = os.path.join(
        settings["outdir"], str(settings["unique_run_id"])
    )

    results_queue = mp.JoinableQueue()
    s = "Initialised the results queue."
    Logger.info(s, timings=True)

    writer_process = mp.Process(
        target=writer_boilerplate_fn,
        args=(results_queue, settings)
    )

    writer_process.start()
    s = "Started a writer subprocess."
    Logger.info(s, timings=True)

    try:

        for batch in batch_gen:

            # the skip_n_batch argument can be used to skip a specified number
            # of batches -- useful if things break in the second batch but not
            # the first!

            if batch_gen.current_idx <= settings["skip_n_batch"]:
                continue

            # we *must* call the tensorflow session within the batch loop so the
            # graph gets reset: the maximum example length in a batch affects
            # the size of most graph elements.

            s = "Running for Batch Number: {b} of {n}".format(
                b=batch_gen.current_idx, n=batch_gen.n_batches
            )
            Logger.info(s, timings=True)

            executor_boilerplate_fn(
                results_queue,
                settings,
                batch,
                attack_fn
            )
            s = "Finished Batch Number: {b} of {n}".format(
                b=batch_gen.current_idx, n=batch_gen.n_batches
            )
            Logger.info(s, timings=True)

    except BaseException as e:

        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        Logger.critical("{e}".format(e=tb), timings=True)

    else:

        Logger.log(
            "Finished processing all batches, run complete.", timings=True
        )

    finally:

        Logger.info(
            "Attempting to close/terminate results queue/writer subprocess.",
            timings=True
        )
        results_queue.put("dead")
        results_queue.close()
        Logger.info("Results queue closed.", timings=True)

        writer_process.join()
        writer_process.terminate()
        Logger.info("Writer subprocess terminated.", timings=True)


def default_manager(settings, attack_fn, batch_gen):

    return manager(
        settings,
        attack_fn,
        batch_gen,
    )

