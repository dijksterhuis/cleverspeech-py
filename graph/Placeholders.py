"""
Placeholder class objects with tensorflow.placeholder objects as attributes. The
standard attack placeholder is the only one currently implemented. This object
can be imported and extended, although best practice would be to create your own
object if you need more placeholders (you'll probably be making other
modifications to classes too).

--------------------------------------------------------------------------------
"""


import tensorflow as tf


class Placeholders(object):
    """
    A simple placeholders object. The attributes in this object are the absolute
    minimum placeholders required to create an attack.

    :param batch_size: the size of the current input batch, a cleverspeech.data.ingress.batch_generators.batch object
    :param maxlen: the maximum length of audio examples, post padding for feature extraction
    """
    def __init__(self, batch):

        batch_size, maxlen = batch.size, batch.audios["max_samples"]

        self.audios = tf.placeholder(tf.float32, [batch_size, maxlen], name="new_input")
        self.audio_lengths = tf.placeholder(tf.int32, [batch_size], name='qq_featlens')
        self.targets = tf.placeholder(tf.int32, [batch_size, None], name='qq_targets')
        self.target_lengths = tf.placeholder(tf.int32, [batch_size], name='qq_target_lengths')
