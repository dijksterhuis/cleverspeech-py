import os
import traceback
import multiprocessing as mp

from progressbar import ProgressBar

from cleverspeech.data.egress import load
from cleverspeech.runtime.TensorflowRuntime import TFRuntime
from cleverspeech.utils.Utils import log


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


def writer_boilerplate_fn(results_transforms, queue, settings):

    import traceback
    from time import sleep

    try:

        settings_writer = settings["writer"].split("_")[0]
        SETTINGS_WRITER_FUNCS[settings_writer](settings["outdir"], settings)

        while True:

            sleep(1)
            results = queue.get()

            if results == "dead":
                queue.task_done()
                raise EndOfRunException

            else:
                for example in results_transforms(results, settings):
                    if example["success"] is True:
                        RESULTS_WRITER_FUNCS[settings["writer"]](
                            settings["outdir"],
                            example
                        )
    except EndOfRunException:
        s = "Dead queue entry detected... Writer subprocess exiting."
        log(s)
        pass

    except KeyboardInterrupt:
        s = "\nKeyboardInterrupt detected... Writer subprocess exiting."
        log(s)

    except BaseException as e:
        tb = "".join(
            traceback.format_exception(None, e, e.__traceback__))

        s = "\033[1;31mSomething broke during file writes!\033[1;0m"
        s += "\n\nError Traceback:\n{e}".format(e=tb)
        log(s)
        raise e


def executor_boilerplate_fn(extract_fn, results_queue, settings, batch, attack_fn):

    # tensorflow sessions can't be passed between processes
    tf_runtime = TFRuntime(settings["gpu_device"])

    with tf_runtime.session as sess, tf_runtime.device as tf_device:

        # Initialise attack graph constructor function
        attack = attack_fn(sess, batch, settings)

        # log some useful things for debugging before the attack runs
        attack.validate()

        s = "Created Attack Graph and Feeds. Loaded TF Operations:"
        log(s, wrap=False)
        log(funcs=tf_runtime.log_attack_tensors)

        s = "Beginning attack run...\nMonitor progress in: {}".format(
            os.path.join(settings["outdir"], "log.txt")
        )
        log(s)

        try:

            with ProgressBar(min_value=1, max_value=settings["nsteps"]) as p:

                for step, is_results_step in attack.run():

                    if is_results_step:
                        res = extract_fn(attack)
                        results_queue.put(res)

                    if step > 0:
                        p.update(step)

        except BaseException:
            log("")
            s = "\033[1;31mAttack failed to run for these examples:\033[1;0m\n"
            s += '\n'.join(batch.audios["basenames"])
            log(s)
            raise


def manager(settings, attack_fn, batch_gen, results_extract_fn=None, results_transform_fn=None):

    if not results_extract_fn:
        from cleverspeech.data.egress.extract import get_attack_state
        results_extract_fn = get_attack_state

    if not results_transform_fn:
        from cleverspeech.data.egress.transform import transforms_gen
        results_transform_fn = transforms_gen

    # modify this as late as possible to catch any added directories in exp defs
    settings["outdir"] = os.path.join(
        settings["outdir"], str(settings["unique_run_id"])
    )

    results_queue = mp.JoinableQueue()
    log("Initialised the results queue.")

    writer_process = mp.Process(
        target=writer_boilerplate_fn,
        args=(results_transform_fn, results_queue, settings)
    )

    writer_process.start()
    log("Started a writer subprocess.")

    try:

        for b_id, batch in batch_gen:

            # we *must* call the tensorflow session within the batch loop so the
            # graph gets reset: the maximum example length in a batch affects
            # the size of most graph elements.

            log("Running for Batch Number: {}".format(b_id), wrap=True)
            executor_boilerplate_fn(
                results_extract_fn,
                results_queue,
                settings, batch,
                attack_fn
            )

    except BaseException as e:

        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        log(
            "\033[1;31mERROR TRACEBACK:\033[1;0m\n{e}".format(e=tb),
            wrap=True
        )

    finally:

        log(
            "Attempting to close/terminate results queue/writer subprocess.",
            wrap=True
        )
        results_queue.put("dead")
        results_queue.close()
        log("Results queue closed.", wrap=True)

        writer_process.join()
        writer_process.terminate()
        log("Writer subprocess terminated.", wrap=True)


def default_manager(settings, attack_fn, batch_gen):

    return manager(
        settings,
        attack_fn,
        batch_gen,
    )

