#!/usr/bin/env python3
import os
import traceback
import tensorflow as tf
import numpy as np

from cleverspeech import data
from cleverspeech import graph
from cleverspeech import models

from cleverspeech.config.arguments_parser import args
from cleverspeech.runtime.TensorflowRuntime import TFRuntime
from cleverspeech.utils.Utils import log


def custom_manager(settings, model_fn, batch_gen):

    n_examples, n_frames = 0, 0
    token_probs, cers, wers = np.asarray([0.0]*29), 0, 0
    tokens = None
    for batch in batch_gen:
        tokens = batch.targets["tokens"]
        try:
            tf_runtime = TFRuntime(settings["gpu_device"])
            with tf_runtime.session as sess, tf_runtime.device as tf_device:

                model = model_fn(sess, batch, settings)

                decodings, probs = model.inference(
                    batch,
                    feed=model.feeds.examples,
                    top_five=False
                )

                logits, softmax = sess.run(
                    [tf.transpose(model.raw_logits, [1, 0, 2]), model.logits],
                    feed_dict=model.feeds.examples,
                )

                outdir = os.path.join(settings["outdir"], "validation")
                if not os.path.exists(outdir):
                    os.makedirs(outdir, exist_ok=True)

                for idx, basename in enumerate(batch.audios["basenames"]):

                    n_examples += 1
                    n_frames += len(softmax[idx])

                    token_probs += np.sum(softmax[idx], axis=0)

                    outpath = os.path.join(
                        outdir, basename.rstrip(".wav") + ".json"
                    )

                    cer = data.metrics.transcription_error.character_error_rate(
                        decodings[idx],
                        batch.trues["true_targets"][idx],
                    )

                    cers += cer

                    wer = data.metrics.transcription_error.word_error_rate(
                        decodings[idx],
                        batch.trues["true_targets"][idx],
                    )

                    wer += wer

                    res = {
                        "decoding": decodings[idx],
                        "probs": probs[idx],
                        "logits": logits[idx],
                        "softmax": softmax[idx],
                        "wer": wer,
                        "cer": cer,
                    }

                    json_res = data.egress.load.prepare_json_data(res)

                    with open(outpath, "w+") as f:
                        f.write(json_res)

                    s = "Sample {i}\t Basename {b}\tProbs {p:.3f}\tWER: {w:.3f}\tCER: {c:.3f}\tdecoding: {d}".format(
                        i=idx, b=basename, p=probs[idx], w=wer, c=cer,
                        d=decodings[idx]
                    )
                    log(s, wrap=False)

        except Exception as e:
            tb = "".join(traceback.format_exception(None, e, e.__traceback__))

            s = "Something broke! Attack failed to run for these examples:\n"
            s += '\n'.join(batch.audios["basenames"])
            s += "\n\nError Traceback:\n{e}".format(e=tb)

            log(s, wrap=True)
            raise

    log("", wrap=True)
    log("Final Summary:", wrap=False)
    s = "Examples: {e}\tFrames: {f}\t Mean WER: {w}\tMean CER: {c}".format(
        e=n_examples, f=n_frames, w=wers/n_examples, c=cers/n_examples
    )
    log(s, wrap=True)
    log("Mean per token probabilities:", wrap=True)
    mean_token_probs = token_probs / n_frames
    for idx, token in enumerate(tokens):
        log("{t} = {s}".format(t=token, s=mean_token_probs[idx]), wrap=False)


def create_validation_graph(sess, batch, settings):

    feeds = graph.GraphConstructor.Feeds()

    audios_ph = tf.placeholder(
        tf.float32, [batch.size, batch.audios["max_samples"]], name="new_input"
    )
    audio_lengths_ph = tf.placeholder(
        tf.int32, [batch.size], name='qq_featlens'
    )

    feeds.examples = {
        audios_ph: batch.audios["padded_audio"],
        audio_lengths_ph: batch.audios["ds_feats"],
    }

    model = models.DeepSpeech.Model(
        sess, audios_ph, batch,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )

    # hacky
    model.feeds = feeds

    return model


def attack_run(master_settings):

    audios = data.ingress.mcv_v1.MCV1StandardAudioBatchETL(
        master_settings["audio_indir"],
        master_settings["max_examples"],
        filter_term=".wav",
        max_file_size=master_settings["max_audio_file_bytes"]
    )

    transcriptions = data.ingress.mcv_v1.MCV1TranscriptionsFromCSVFile(
        master_settings["targets_path"],
        master_settings["max_targets"],
    )

    batch_gen = data.ingress.mcv_v1.MCV1IterableBatches(
        master_settings, audios, transcriptions
    )

    master_settings["outdir"] = os.path.join(
        master_settings["outdir"], master_settings["decoder"]
    )

    custom_manager(
        master_settings,
        create_validation_graph,
        batch_gen,
    )

    log("Finished all runs.")


def main():
    log("", wrap=True)
    args(attack_run)


if __name__ == '__main__':
    main()


