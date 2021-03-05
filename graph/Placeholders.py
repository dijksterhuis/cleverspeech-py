import tensorflow as tf


class Placeholders(object):
    def __init__(self, batch_size, maxlen):
        self.audios = tf.placeholder(tf.float32, [batch_size, maxlen], name="new_input")
        self.audio_lengths = tf.placeholder(tf.int32, [batch_size], name='qq_featlens')
        self.targets = tf.placeholder(tf.int32, [batch_size, None], name='qq_targets')
        self.target_lengths = tf.placeholder(tf.int32, [batch_size], name='qq_target_lengths')
