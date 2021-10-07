import tensorflow as tf
import numpy as np
import os
import multiprocessing as mp
import ds_ctcdecoder

from abc import ABC
from multiprocessing import cpu_count
from cleverspeech.utils.Utils import lcomp, l_map

from cleverspeech.models.__DeepSpeech_v0_9_3.src.training.deepspeech_training.train import create_model
from cleverspeech.models.__DeepSpeech_v0_9_3.src.training.deepspeech_training.util.config import Config, initialize_globals
from cleverspeech.models.__DeepSpeech_v0_9_3.src.training.deepspeech_training.util.flags import create_flags


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
    def __init__(self, sess, input_tensor, batch, beam_width=500, decoder='ds', tokens=" abcdefghijklmnopqrstuvwxyz'-"):

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
        self.decoder = decoder
        self.beam_width = beam_width
        # beam_width = tf.app.flags.FLAGS.beam_width

        self.raw_logits = None
        self.logits = None
        self.inputs = None
        self.outputs = None
        self.layers = None
        self.__reset_rnn_state = None
        self.saver = None
        self.alphabet = None
        self.scorer = None

        self.initialise()
        self.create_graph(input_tensor, batch.audios["ds_feats"], batch.size)
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

        self.alphabet = ds_ctcdecoder.Alphabet(
            os.path.abspath(tf.app.flags.FLAGS.alphabet_config_path))

        self.scorer = ds_ctcdecoder.Scorer(
            tf.app.flags.FLAGS.lm_alpha,
            tf.app.flags.FLAGS.lm_beta,
            tf.app.flags.FLAGS.scorer_path,
            self.alphabet,
        )

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

    def create_graph(self, input_tensor, seq_length, batch_size):

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
        self.reset_state()

        self.raw_logits = layers['raw_logits']
        self.logits = tf.nn.softmax(tf.transpose(self.raw_logits, [1, 0, 2]))

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
        pass
        #self.sess.run(self.__reset_rnn_state)

    def tf_run(self, *args, **kwargs):
        self.reset_state()
        outs = self.sess.run(*args, **kwargs)
        self.reset_state()
        return outs

    def inference(self, batch, feed=None, logits=None, decoder=None, top_five=False):

        if decoder:
            decoder = decoder
        else:
            decoder = self.decoder

        if decoder == "tf_beam":

            if top_five is True:
                raise NotImplementedError(
                    "top_five is not implemented for the tf decoder"
                )

            if logits is None:
                logits = self.get_logits(self.raw_logits, feed)

            decodings = self.tf_beam_decode(
                logits,
                batch.audios["ds_feats"],
                self.tokens,
            )
            return decodings

        elif decoder == "ds" or not decoder:

            if logits is None:
                logits = self.get_logits(self.logits, feed)

            decoded_with_probs = l_map(
                lambda x: self.ds_decode(x, top_five=top_five), logits
            )

            # TODO

            decodings = l_map(lambda x: x[0], decoded_with_probs)
            probs = l_map(lambda x: x[1], decoded_with_probs)
            token_order = l_map(lambda x: x[2], decoded_with_probs)
            timsteps = l_map(lambda x: x[3], decoded_with_probs)

            return decodings, probs

        elif decoder == "batch":

            if logits is None:
                logits = self.get_logits(self.logits, feed)

            decodings, probs = self.ds_decode_batch(
                logits,
                batch.audios["ds_feats"],
                top_five=True,
            )

            if not top_five:
                decodings = l_map(lambda x: x[0], decodings)
                probs = l_map(lambda x: x[0], probs)

            return decodings, probs

        elif decoder == "batch_no_lm":

            if logits is None:
                logits = self.get_logits(self.logits, feed)

            decodings, probs = self.ds_decode_batch_no_lm(
                logits,
                batch.audios["ds_feats"],
                top_five=True,
            )

            if not top_five:
                decodings = l_map(lambda x: x[0], decodings)
                probs = l_map(lambda x: x[0], probs)

            return decodings, probs

        elif decoder == "greedy_no_lm" or decoder == "greedy":

            if logits is None:
                logits = self.get_logits(self.logits, feed)

            decodings, probs = self.ds_decode_batch_greedy_no_lm(
                logits,
                batch.audios["ds_feats"],
                top_five=False,
            )

            decodings = l_map(lambda x: x[0], decodings)
            probs = l_map(lambda x: x[0], probs)

            return decodings, probs

        elif decoder == "tf_greedy":

            if top_five is True:
                raise NotImplementedError(
                    "top_five is not implemented for greedy decoders"
                )

            if logits is None:
                logits = self.get_logits(self.logits, feed)

            if type(logits) == np.ndarray and logits.shape[0] == batch.size:
                # batch major but tf greedy search wants time major
                logits = np.transpose(logits, [1, 0, 2])

            elif type(logits) == tf.Tensor and logits.get_shape().as_list()[
                0] == batch.size:
                # batch major but tf greedy search wants time major
                logits = tf.transpose(logits, [1, 0, 2])

            else:
                pass

            decodings = self.tf_greedy_decode(
                logits,
                batch.audios["ds_feats"],
                self.tokens,
            )
            return decodings
        elif decoder == "hotfix_greedy":

            if top_five is True:
                raise NotImplementedError(
                    "top_five is not implemented for greedy decoders"
                )

            if logits is None:
                logits = self.get_logits(self.logits, feed)

            if type(logits) == np.ndarray and logits.shape[0] != batch.size:
                # time major but hotfix greedy search wants batch major
                logits = np.transpose(logits, [1, 0, 2])

            elif type(logits) == tf.Tensor and logits.get_shape().as_list()[0] != batch.size:
                # time major but hotfix greedy search wants batch major
                logits = tf.transpose(logits, [1, 0, 2])

            decodings = self.hotfix_greedy_decode(logits, self.tokens)
            return decodings

        else:
            raise Exception(
                "Please choose a valid decoder."
            )

    def ds_decode(self, logits, top_five=False, with_metadata=False):

        decoded_probs = ds_ctcdecoder.swigwrapper.ctc_beam_search_decoder(
            np.squeeze(logits),
            Config.alphabet,
            self.beam_width,
            1,
            40,
            self.scorer,
            dict(),
            1 if not top_five else 5
        )

        beam_results = l_map(
            lambda x:
            (
                Config.alphabet.Decode(x.tokens),
                x.confidence,
                lcomp(x.tokens),
                lcomp(x.timesteps)
            ),
            decoded_probs
        )

        labellings = l_map(
            lambda x: x[0], beam_results
        )

        probs = l_map(
            lambda x: x[1], beam_results
        )

        token_order = l_map(
            lambda x: x[2], beam_results
        )

        timestep_switches = l_map(
            lambda x: x[3], beam_results
        )
        if with_metadata:
            return labellings, probs, token_order, timestep_switches
        else:
            return labellings, probs

    def ds_decode_batch(self, logits, lengths, top_five=False, with_metadata=False):

        l = lengths[0]

        # I have 6 cores on my development machine -- I also want to do other
        # things like write papers when running experiments.

        batched_beam_results = ds_ctcdecoder.swigwrapper.ctc_beam_search_decoder_batch(
            logits,
            np.asarray([l for _ in range(logits.shape[0])], dtype=np.int32),
            Config.alphabet,
            self.beam_width,
            cpu_count() - 2 if cpu_count() > 2 else 1,
            1,
            40,
            self.scorer,
            dict(),
            1 if not top_five else 5,
        )

        beam_results = [l_map(
            lambda x:
            (
                Config.alphabet.Decode(x.tokens),
                x.confidence,
                lcomp(x.tokens),
                lcomp(x.timesteps)
            ),
            beam_results
        ) for beam_results in batched_beam_results]

        labellings = l_map(
            lambda y: l_map(lambda x: x[0], y), beam_results
        )

        probs = l_map(
            lambda y: l_map(lambda x: x[1], y), beam_results
        )

        token_order = l_map(
            lambda y: l_map(lambda x: x[2], y), beam_results
        )

        timestep_switches = l_map(
            lambda y: l_map(lambda x: x[3], y), beam_results
        )

        if with_metadata:
            return labellings, probs, token_order, timestep_switches
        else:
            return labellings, probs

    def ds_decode_batch_greedy_no_lm(self, logits, lengths, top_five=False, with_metadata=False):

        l = lengths[0]

        # I have 6 cores on my development machine -- I also want to do other
        # things like write papers when running experiments.

        batched_beam_results = ds_ctcdecoder.swigwrapper.ctc_beam_search_decoder_batch(
            logits,
            np.asarray([l for _ in range(logits.shape[0])], dtype=np.int32),
            Config.alphabet,
            1,
            cpu_count() - 2 if cpu_count() > 2 else 1,
            1,
            1,
            None,
            dict(),
            1 if not top_five else 5,
        )

        beam_results = [l_map(
            lambda x:
            (
                Config.alphabet.Decode(x.tokens),
                x.confidence,
                lcomp(x.tokens),
                lcomp(x.timesteps)
            ),
            beam_results
        ) for beam_results in batched_beam_results]

        labellings = l_map(
            lambda y: l_map(lambda x: x[0], y), beam_results
        )

        probs = l_map(
            lambda y: l_map(lambda x: x[1], y), beam_results
        )

        token_order = l_map(
            lambda y: l_map(lambda x: x[2], y), beam_results
        )

        timestep_switches = l_map(
            lambda y: l_map(lambda x: x[3], y), beam_results
        )

        if with_metadata:
            return labellings, probs, token_order, timestep_switches
        else:
            return labellings, probs

    def ds_decode_batch_no_lm(self, logits, lengths, top_five=False, with_metadata=False):

        l = lengths[0]

        # I have 6 cores on my development machine -- I also want to do other
        # things like write papers when running experiments.

        batched_beam_results = ds_ctcdecoder.swigwrapper.ctc_beam_search_decoder_batch(
            logits,
            np.asarray([l for _ in range(logits.shape[0])], dtype=np.int32),
            Config.alphabet,
            self.beam_width,
            cpu_count() - 2 if cpu_count() > 2 else 1,
            1,
            40,
            None,
            dict(),
            1 if not top_five else 5,
        )

        beam_results = [l_map(
            lambda x:
            (
                Config.alphabet.Decode(x.tokens),
                x.confidence,
                lcomp(x.tokens),
                lcomp(x.timesteps)
            ),
            beam_results
        ) for beam_results in batched_beam_results]

        labellings = l_map(
            lambda y: l_map(lambda x: x[0], y), beam_results
        )

        probs = l_map(
            lambda y: l_map(lambda x: x[1], y), beam_results
        )

        token_order = l_map(
            lambda y: l_map(lambda x: x[2], y), beam_results
        )

        timestep_switches = l_map(
            lambda y: l_map(lambda x: x[3], y), beam_results
        )

        if with_metadata:
            return labellings, probs, token_order, timestep_switches
        else:
            return labellings, probs

    def tf_beam_decode(self, logits, features_lengths, tokens):

        tf_decode, log_probs = tf.nn.ctc_beam_search_decoder(
            logits,
            features_lengths,
            merge_repeated=False,
            beam_width=self.beam_width
        )
        dense = [tf.sparse.to_dense(tf_decode[0])]
        tf_dense, probs = self.tf_run([dense, log_probs])

        tf_outputs = l_map(
            lambda y: ''.join([tokens[int(x)] for x in y]),
            tf_dense[0]
        )

        tf_outputs = [o.rstrip(" ") for o in tf_outputs]
        probs = [prob[0] for prob in probs]

        return tf_outputs, probs

    def tf_greedy_decode(self, logits, features_lengths, tokens, merge_repeated=True):

        tf_decode, log_probs = tf.nn.ctc_greedy_decoder(
            logits,
            features_lengths,
            merge_repeated=merge_repeated,
        )
        dense = [tf.sparse.to_dense(tf_decode[0])]

        tf_dense, neg_sum_logits = self.tf_run([dense, log_probs])

        tf_outputs = l_map(
            lambda y: ''.join([tokens[int(x)] for x in y]),
            tf_dense[0]
        )

        tf_outputs = [o.rstrip(" ") for o in tf_outputs]
        neg_sum_logits = [prob[0] for prob in neg_sum_logits]

        return tf_outputs, neg_sum_logits

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