import tensorflow as tf
import numpy as np
import sys

from cleverspeech.data import Batches
from cleverspeech.data import Generators
#from SecEval import VictimAPI as Victim
from cleverspeech.utils.Utils import log
from cleverspeech.utils.RuntimeUtils import create_tf_runtime


def main(indir, target, batch_size, tokens=" abcdefghijklmnopqrstuvwxyz'-"):

    # Create the factory we'll use to iterate over N examples at a time.

    batch_factory = Generators.BatchGenerator(
        indir,
        None,
        target_phrase=target,
        tokens=tokens,
        sort_by_file_size="desc",
        filter_term="advex"
    )

    batch_gen = batch_factory.generate(
        Batches.ValidationLoader,
        batch_size=batch_size
    )

    for b_id, batch in batch_gen:

        tf_session, tf_device = create_tf_runtime()
        with tf_session as sess, tf_device:

            ph_examples = tf.placeholder(tf.float32, shape=[batch.size, batch.audios.max_length])
            ph_lens = tf.placeholder(tf.float32, shape=[batch.size])

            model = DeepSpeech.Model(sess, tf.ceil(ph_examples), batch, tokens=tokens)

            batch.feeds.create_feeds(ph_examples, ph_lens)

            s = "{}:\t".format(batch.audios.basenames[0])

            decodings, probs = model.inference(
                batch,
                feed=batch.feeds.examples,
                decoder="batch",
                beam_width=500,
                top_five=True
            )

            s += "Prob: {:.3f}\t".format(probs[0][0])
            s += "Decoding: {}".format(decodings[0][0])

            log(s, wrap=False)


if __name__ == '__main__':
    indir, target = sys.argv[1:]

    main(indir, target, int(1))

