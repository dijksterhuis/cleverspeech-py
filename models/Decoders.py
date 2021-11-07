import tensorflow as tf
import numpy as np
import os
import ds_ctcdecoder

from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from cleverspeech.utils.Utils import lcomp, l_map
from cleverspeech.models.__DeepSpeech_v0_9_3.src.training.deepspeech_training.util.config import Config


class _AbstractDecoder(ABC):
    def __init__(self, tf_sess, tf_logits, batch, feed, tokens, *args, **kwargs):

        self.tokens = tokens
        self._feed = feed
        self._tf_sess = tf_sess
        self._batch = batch
        self._tf_logits = tf_logits

    @abstractmethod
    def get_logits(self, *args, **kwargs):
        pass

    @abstractmethod
    def inference(self, top_five=False):
        pass

    @abstractmethod
    def _decoding(self, *args, **kwargs):
        pass


class _BeamSearchDecoder(_AbstractDecoder):
    def __init__(self, tf_sess, tf_logits, batch, feed, tokens, beam_width):
        super().__init__(tf_sess, tf_logits, batch, feed, tokens)
        self.beam_width = beam_width


class _GreedySearchDecoder(_AbstractDecoder):
    def __init__(self, tf_sess, tf_logits, batch, feed, tokens):
        super().__init__(tf_sess, tf_logits, batch, feed, tokens)
        self.beam_width = 1


class _DeepSpeechDecoder(_AbstractDecoder):
    def get_logits(self, logits):
        return self._tf_sess.run(logits, feed_dict=self._feed)

    @staticmethod
    def _convert_beam_to_lists(batched_beam_results):
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
        return labellings, probs, token_order, timestep_switches


class _TensorflowDecoder(_AbstractDecoder):

    def get_logits(self, logits, time_major=False):

        if not time_major and logits.shape[0] != self._batch.size:
            return tf.transpose(logits, [1, 0, 2])

        if time_major and logits.shape[0] == self._batch.size:
            return tf.transpose(logits, [1, 0, 2])

    def inference(self, top_five=False):

        if top_five is True:
            raise NotImplementedError(
                "top_five is not implemented for Tensorflow decoders"
            )

        decodings = self._decoding(
            self._tf_logits,
            self._batch.audios["ds_feats"],
            self.tokens,
            self._feed
        )
        return decodings


class _DeepSpeechBeamSearchMixin(_DeepSpeechDecoder, _BeamSearchDecoder):
    pass


class DeepSpeechBeamSearchWithoutLanguageModelBatchDecoder(_DeepSpeechBeamSearchMixin):

    def __init__(self, tf_sess, tf_logits, batch, feed, tokens, beam_width):
        super().__init__(tf_sess, tf_logits, batch, feed, tokens, beam_width)

        self.alphabet = ds_ctcdecoder.Alphabet(
            os.path.abspath(tf.app.flags.FLAGS.alphabet_config_path))

        self.scorer = None

        self.decode_op = ds_ctcdecoder.swigwrapper.ctc_beam_search_decoder_batch

    def inference(self, top_five=False):

        logits = self.get_logits(self._tf_logits)

        decodings, probs = self._decoding(
            logits,
            self._batch.audios["ds_feats"],
            top_five=True,
        )

        if not top_five:
            decodings = l_map(lambda x: x[0], decodings)
            probs = l_map(lambda x: x[0], probs)

        return decodings, probs

    def _decoding(self, logits, lengths, top_five=False, with_metadata=False):

        feature_lengths = np.asarray(
            [lengths[0] for _ in range(logits.shape[0])],
            dtype=np.int32
        )

        # I have 6 cores on my development machine -- I also want to do other
        # things like write papers when running experiments.
        n_processes = cpu_count() - 2 if cpu_count() > 2 else 1
        n_results = 1 if not top_five else 5
        n_cutoff_top = 40
        cutoff_top_prob = 1

        batched_beam_results = self.decode_op(
            logits,
            feature_lengths,
            Config.alphabet,
            self.beam_width,
            n_processes,
            cutoff_top_prob,
            n_cutoff_top,
            self.scorer,
            dict(),
            n_results,
        )

        (
            labellings,
            probs,
            token_order,
            timestep_switches
        ) = self._convert_beam_to_lists(batched_beam_results)

        if with_metadata:
            return labellings, probs, token_order, timestep_switches
        else:
            return labellings, probs


class DeepSpeechBeamSearchWithLanguageModelBatchDecoder(
    DeepSpeechBeamSearchWithoutLanguageModelBatchDecoder
):
    def __init__(self, tf_sess, tf_logits, batch, feed, tokens, beam_width):
        super().__init__(tf_sess, tf_logits, batch, feed, tokens, beam_width)

        self.scorer = ds_ctcdecoder.Scorer(
            tf.app.flags.FLAGS.lm_alpha,
            tf.app.flags.FLAGS.lm_beta,
            tf.app.flags.FLAGS.scorer_path,
            self.alphabet,
        )


class DeepSpeechBeamSearchWithoutLanguageModelDecoder(_DeepSpeechBeamSearchMixin):

    def __init__(self, tf_sess, tf_logits, batch, feed, tokens, beam_width):
        super().__init__(tf_sess, tf_logits, batch, feed, tokens, beam_width)

        self.alphabet = ds_ctcdecoder.Alphabet(
            os.path.abspath(tf.app.flags.FLAGS.alphabet_config_path))

        self.scorer = None

        self.decode_op = ds_ctcdecoder.swigwrapper.ctc_beam_search_decoder

    def inference(self, top_five=False):

        logits = self.get_logits(self._tf_logits)

        decodings, probs = self._decoding(
            logits,
            self._batch.audios["ds_feats"],
            top_five=True,
        )

        if not top_five:
            decodings = l_map(lambda x: x[0], decodings)
            probs = l_map(lambda x: x[0], probs)

        return decodings, probs

    def _decoding(self, logits, lengths, top_five=False, with_metadata=False):

        n_results = 1 if not top_five else 5
        n_cutoff_top = 40
        cutoff_top_prob = 1

        batched_beam_results = self.decode_op(
            np.squeeze(logits),
            Config.alphabet,
            self.beam_width,
            cutoff_top_prob,
            n_cutoff_top,
            self.scorer,
            dict(),
            n_results,
        )

        (
            labellings,
            probs,
            token_order,
            timestep_switches
        ) = self._convert_beam_to_lists(batched_beam_results)

        if with_metadata:
            return labellings, probs, token_order, timestep_switches
        else:
            return labellings, probs


class DeepSpeechBeamSearchWithLanguageModelDecoder(
    DeepSpeechBeamSearchWithoutLanguageModelDecoder
):
    def __init__(self, tf_sess, tf_logits, batch, feed, tokens, beam_width):
        super().__init__(tf_sess, tf_logits, batch, feed, tokens, beam_width)

        self.scorer = ds_ctcdecoder.Scorer(
            tf.app.flags.FLAGS.lm_alpha,
            tf.app.flags.FLAGS.lm_beta,
            tf.app.flags.FLAGS.scorer_path,
            self.alphabet,
        )


class TensorflowBeamSearchWithoutLanguageModelDecoder(
    _BeamSearchDecoder, _TensorflowDecoder
):
    def __init__(self, tf_sess, tf_logits, batch, feed, tokens, beam_width):
        super().__init__(tf_sess, tf_logits, batch, feed, tokens, beam_width)

        logits = self.get_logits(self._tf_logits, time_major=False)

        tf_decode, log_probs = tf.nn.ctc_beam_search_decoder(
            logits,
            batch.audios["ds_feats"],
            merge_repeated=False,
            beam_width=self.beam_width
        )
        dense = [tf.sparse.to_dense(tf_decode[0])]

        self.decode_op = [dense, log_probs]

    def _decoding(self, logits, lengths, top_five=False, with_metadata=False):

        tf_dense, neg_sum_logits = self._tf_sess.run(self.decode_op, self._feed)

        tf_outputs = l_map(
            lambda y: ''.join([self.tokens[int(x)] for x in y]),
            tf_dense[0]
        )

        tf_outputs = [o.rstrip(" ") for o in tf_outputs]
        neg_sum_logits = [prob[0] for prob in neg_sum_logits]

        return tf_outputs, neg_sum_logits


class TensorflowGreedySearchWithoutLanguageModelDecoder(
    _BeamSearchDecoder, _TensorflowDecoder
):
    def __init__(self, tf_sess, tf_logits, batch, feed, tokens, beam_width):
        super().__init__(tf_sess, tf_logits, batch, feed, tokens, beam_width)

        logits = self.get_logits(self._tf_logits, time_major=True)

        tf_decode, log_probs = tf.nn.ctc_greedy_decoder(
            logits,
            batch.audios["ds_feats"],
            merge_repeated=True,
        )
        dense = [tf.sparse.to_dense(tf_decode[0])]

        self.decode_op = [dense, log_probs]

        print(self.decode_op)

    def _decoding(self, logits, lengths, top_five=False, with_metadata=False):
        tf_dense, neg_sum_logits = self._tf_sess.run(self.decode_op, self._feed)

        tf_outputs = l_map(
            lambda y: ''.join([self.tokens[int(x)] for x in y]),
            tf_dense[0]
        )

        tf_outputs = [o.rstrip(" ") for o in tf_outputs]
        neg_sum_logits = [prob[0] for prob in neg_sum_logits]

        return tf_outputs, neg_sum_logits
