import traceback
import multiprocessing as mp

from cleverspeech.data.egress import extract, load
from cleverspeech.utils.runtime.TensorflowRuntime import TFRuntime
from cleverspeech.utils.Utils import log


RESULTS_WRITER_FUNCS = {
    "local_latest": load.write_latest_metadata_to_local_json_file,
    "local_all": load.write_per_bound_metadata_to_local_json_files,
    "s3_latest": load.write_latest_metadata_to_s3,
    "s3_all": load.write_per_bound_metadata_to_local_json_files,
}

SETTINGS_WRITER_FUNCS = {
    "local": load.write_settings_to_local_json_file,
    "s3": load.write_settings_to_s3,
}


def writer_boilerplate_fn(results_transforms, queue, settings):

    import traceback

    settings_writer = settings["writer"].split("_")[0]
    SETTINGS_WRITER_FUNCS[settings_writer](settings["outdir"], settings)

    while True:
        results = queue.get()

        if results == "dead":
            queue.task_done()
            break

        else:
            try:
                for example in results_transforms(results, settings):
                    if example["success"] is True:
                        RESULTS_WRITER_FUNCS[settings["writer"]](
                            settings["outdir"],
                            example
                        )

            except Exception as e:
                tb = "".join(
                    traceback.format_exception(None, e, e.__traceback__))

                s = "Something broke during file writes!"
                s += "\n\nError Traceback:\n{e}".format(e=tb)
                log(s, wrap=True)
                raise

            finally:
                queue.task_done()


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
            settings["outdir"] + "log.txt"
        )
        log(s)

        for is_results_step in attack.run():
            if is_results_step:
                res = extract_fn(attack)
                results_queue.put(res)


def manager(settings, attack_fn, batch_gen, results_extract_fn, results_transform_fn):

    results_queue = mp.JoinableQueue()

    writer_process = mp.Process(
        target=writer_boilerplate_fn,
        args=(results_transform_fn, results_queue, settings)
    )

    writer_process.start()
    log("Started a writer subprocess.")

    for b_id, batch in batch_gen:

        # we *must* call the tensorflow session within the batch loop so the
        # graph gets reset: the maximum example length in a batch affects the
        # size of most graph elements.

        log("Running for Batch Number: {}".format(b_id), wrap=True)

        attack_process = mp.Process(
            target=executor_boilerplate_fn,
            args=(results_extract_fn, results_queue, settings, batch, attack_fn)
        )

        try:
            attack_process.start()
            attack_process.join()
            attack_process.terminate()

        except Exception as e:

            tb = "".join(traceback.format_exception(None, e, e.__traceback__))

            s = "Something broke! Attack failed to run for these examples:\n"
            s += '\n'.join(batch.audios["basenames"])
            s += "\n\nError Traceback:\n{e}".format(e=tb)

            log(s, wrap=True)
            log("Attempting to close writer queue and subprocess.", wrap=True)

            results_queue.put("dead")
            results_queue.close()
            log("Results queue closed.", wrap=True)

            writer_process.join()
            writer_process.terminate()
            log("Writer subprocess closed.", wrap=True)
            raise

    log("Attempting to close writer queue and subprocess.", wrap=True)
    results_queue.put("dead")
    results_queue.close()
    log("Results queue closed.", wrap=True)

    writer_process.join()
    writer_process.terminate()
    log("Writer subprocess closed.", wrap=True)


def default_unbounded_manager(settings, attack_fn, batch_gen):

    from cleverspeech.data.egress import extract, transform

    return manager(
        settings,
        attack_fn,
        batch_gen,
        extract.get_unbounded_attack_state,
        transform.unbounded_gen
    )


def default_evasion_manager(settings, attack_fn, batch_gen):

    from cleverspeech.data.egress import extract, transform

    return manager(
        settings,
        attack_fn,
        batch_gen,
        extract.get_evasion_attack_state,
        transform.evasion_gen
    )

