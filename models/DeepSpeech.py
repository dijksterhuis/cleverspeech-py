import tensorflow as tf
import numpy as np
import os
import multiprocessing as mp
import ds_ctcdecoder

from abc import ABC
from multiprocessing import cpu_count
from cleverspeech.utils.Utils import lcomp, l_map

from cleverspeech.models.__DeepSpeech_v0_9_3.src.training.deepspeech_training.train import (
  create_inference_graph,
  create_model
)
from cleverspeech.models.__DeepSpeech_v0_9_3.src.training.deepspeech_training.util.config import Config, initialize_globals
from cleverspeech.models.__DeepSpeech_v0_9_3.src.training.deepspeech_training.util.flags import create_flags

from cleverspeech.models.Decoders import (
    DeepSpeechBeamSearchWithoutLanguageModelBatchDecoder,
    DeepSpeechBeamSearchWithLanguageModelBatchDecoder,
    DeepSpeechBeamSearchWithoutLanguageModelDecoder,
    DeepSpeechBeamSearchWithLanguageModelDecoder,
    TensorflowBeamSearchWithoutLanguageModelDecoder,
    TensorflowGreedySearchWithoutLanguageModelDecoder,
)


class TFSignalMFCC:
    def __init__(self, audio_tensor, batch_size, sample_rate=16000, n_context=9, n_ceps=26, cep_lift=22):
        """
        Carlini & Wagners implementation of MFCC & windowing in tensorflow
        :param audio_tensor: the input audio tensor/variable
        :param audio_data: the DataLoader.AudioData test_data class
        :param batch_size: the size of the test data batch
        :param sample_rate: sample rate of the input audio files
        """
        self.window_size = int(0.032 * sample_rate)
        self.window_step = int(0.02 * sample_rate)
        self.sample_rate = sample_rate
        self.n_ceps = n_ceps

        self.audio_input = audio_tensor
        self.batch_size = batch_size

        self.mfcc = None
        self.windowed = None
        self.features = None
        self.features_shape = None
        self.features_len = None

    def mfcc_ops(self):

        stfts = tf.signal.stft(
            self.audio_input,
            frame_length=self.window_size,
            frame_step=self.window_step,
            fft_length=512  # TODO: fft_length setting?!
        )

        spectrograms = tf.abs(stfts)

        num_spectrogram_bins = stfts.shape[-1].value

        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20, 8000, 26

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins,
            num_spectrogram_bins,
            self.sample_rate,
            lower_edge_hertz,
            upper_edge_hertz
        )
        mfcc = tf.tensordot(
            spectrograms,
            linear_to_mel_weight_matrix,
            1
        )

        mfcc.set_shape(
            spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )

        log_mel_spectrograms = tf.math.log(mfcc + 1e-6)

        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :self.n_ceps]

        self.mfcc = tf.reshape(mfcc, [self.batch_size, -1, self.n_ceps])

    def window_ops(self):
        if self.mfcc is None:
            raise AttributeError("You haven't created an mfcc value yet!")
        # Later versions of DeepSpeech / coqui-ai handle windowing within the
        # training graph
        self.features = self.mfcc


class Model(ABC):
    def __init__(self, sess, input_tensor, batch, feed, beam_width=500, decoder='ds', tokens=" abcdefghijklmnopqrstuvwxyz'-"):

        self.sess = sess

        # Add DS lib to the path then run the configuration for it
        model_data_path = os.path.abspath(os.path.dirname(__file__))

        self.checkpoint_dir = os.path.abspath(
            os.path.join(
                model_data_path, "./__DeepSpeech_v0_9_3/data/deepspeech-0.9.3-checkpoint/"
            )
        )
        self.model_dir = os.path.abspath(
            os.path.join(
                model_data_path, "./__DeepSpeech_v0_9_3/data/models/"
            )
        )

        self.tokens = tokens
        self.beam_width = beam_width
        # beam_width = tf.app.flags.FLAGS.beam_width

        self.raw_logits = None
        self.logits = None
        self.inputs = None
        self.outputs = None
        self.layers = None
        self.__reset_rnn_state = None
        self.saver = None
        self.decoder = None
        self.alphabet = None
        self.scorer = None

        self.initialise()
        self.create_graph(input_tensor, batch, feed, decoder)
        self.load_checkpoint()

    def __configure(self):

        create_flags()

        tf.app.flags.FLAGS.alphabet_config_path = os.path.join(
            self.model_dir,
            "alphabet.txt"
        )
        tf.app.flags.FLAGS.scorer_path = os.path.join(
            self.model_dir,
            "deepspeech-0.9.3-models.scorer"
        )
        # tf.app.flags.FLAGS.lm_binary_path = os.path.join(
        #     self.model_dir,
        #     "lm.binary"
        # )
        # tf.app.flags.FLAGS.lm_trie_path = os.path.join(
        #     self.model_dir,
        #     "trie"
        # )
        tf.app.flags.FLAGS.n_steps = -1
        # tf.app.flags.FLAGS.use_seq_length = True

        initialize_globals()

    def initialise(self):
        """
        Check to see if we've already initialised DeepSpeech.
        If not, create the flags and init global DS variables.
        Otherwise, delete all the flags and do the same.

        If we don't do this then the previous state is passed between different
        batches and can lead to weird issues like low decoder confidence scores
        or misspellings in transcriptions.
        """
        for name in list(tf.app.flags.FLAGS):
            delattr(tf.app.flags.FLAGS, name)
        try:
            # does a flag value currently exist?
            assert tf.app.flags.FLAGS.train is not None
        except AttributeError:
            # no
            pass

        else:
            # yes -- see this comment:
            # https://github.com/abseil/abseil-py/issues/36#issuecomment-362367370
            for name in list(tf.app.flags.FLAGS):
                delattr(tf.app.flags.FLAGS, name)

        finally:
            self.__configure()

    def create_graph(self, input_tensor, batch, feed, decoder):

        seq_length, batch_size, = batch.audios["ds_feats"], batch.size

        feature_extraction = TFSignalMFCC(
            input_tensor,
            batch_size
        )
        feature_extraction.mfcc_ops()
        feature_extraction.window_ops()

        outputs, layers = create_model(
            feature_extraction.features,
            seq_length,
            dropout=[None] * 6,
            batch_size=batch_size,
            overlap=True
        )

        self.outputs = outputs
        self.layers = layers

        #self.__reset_rnn_state = outputs["initialize_state"]
        #self.reset_state()

        self.raw_logits = layers['raw_logits']
        self.logits = tf.nn.softmax(tf.transpose(self.raw_logits, [1, 0, 2]))

        self.decoder = TensorflowGreedySearchWithoutLanguageModelDecoder(
            self.sess, self.logits, batch, feed, self.tokens, self.beam_width
        )

    def load_checkpoint(self):

        mapping = {
            v.op.name: v
            for v in tf.global_variables()
            if not v.op.name.startswith('previous_state_')
            and not v.op.name.startswith("qq")
        }

        # print("=-=-=-=- Initialised Variables")
        # for v in tf.global_variables(): print(v)

        self.saver = saver = tf.train.Saver(mapping)
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)

        if not checkpoint:
            raise Exception(
                'Not a valid checkpoint directory ({})'.format(self.checkpoint_dir)
            )

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(self.sess, checkpoint_path)

    def get_logits(self, logits, feed):
        try:
            assert feed is not None
        except AssertionError as e:
            print("You're trying to `get_logits` without providing a feed: {e}".format(e=e))
        else:
            result = self.tf_run(logits, feed_dict=feed)
            return result

    def reset_state(self):
        # self.sess.run(self.__reset_rnn_state)
        pass

    def tf_run(self, *args, **kwargs):
        self.reset_state()
        outs = self.sess.run(*args, **kwargs)
        self.reset_state()
        return outs

    def inference(self, batch, feed=None, logits=None, decoder=None, top_five=False):
        return self.decoder.inference(top_five=top_five)

    @staticmethod
    def reduce(argmax):
        indexes = []
        previous = None
        for current in argmax:

            if current == previous:
                pass
            else:
                indexes.append(current)

            previous = current

        return np.asarray(indexes)

    @staticmethod
    def merge(reduced):
        return reduced[reduced != 28]

    def hotfix_greedy_decode(self, logits, tokens):

        argmaxes = self.tf_run(tf.argmax(logits, axis=-1)).tolist()

        with mp.Pool(cpu_count()) as pool:
            r = pool.map(self.reduce, argmaxes)
            m = pool.map(self.merge, r)

        tf_outputs = l_map(
            lambda y: ''.join([tokens[int(x)] for x in y]).rstrip(" "),
            m
        )

        neg_sum_logits = [0] * logits.shape[0]

        return tf_outputs, neg_sum_logits
