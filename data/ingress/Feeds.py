"""
Mappings of batched input data (e.g. original wav example) to Placeholders. Used
by tensorflow sessions to run operations or to return the current attack state
etc.

The original Carlini & Wagner code defined everything as a variable. We focus on
the traditional tensorflow v1.x placeholder methodology, with attributes of a
Feed object being the inputs for the `feed_dict` in a `sess.run` call.
--------------------------------------------------------------------------------
"""


class Attack(object):
    def __init__(self, batch):
        """
        Holds the feeds which will be passed into DeepSpeech for normal or
        attack evaluation.

        :param audio_batch: a batch of audio examples (`Audios` class)
        :param target_batch: a batch of target phrases (`Targets` class)
        """

        self.audio = batch.audios
        self.targets = batch.targets
        self.examples = None
        self.attack = None
        self.alignments = None

    def create_feeds(self, graph):
        """
        Create the actual feed references from an attack graph. The attack
        graph does not need to be completely constructed yet. We could add a
        VariableGraph and then create the feeds.

        :param graph: the attack graph which holds the input placeholders
        :return: feed_dict for both plain examples and attack optimisation
        """
        # TODO - this is nasty!
        self.alignments = {
            graph.placeholders.targets: self.targets["indices"],
            graph.placeholders.target_lengths: self.targets["lengths"],
        }

        self.examples = {
            graph.placeholders.audios: self.audio["padded_audio"],
            graph.placeholders.audio_lengths: self.audio["ds_feats"],
        }

        self.attack = {
            graph.placeholders.audios: self.audio["padded_audio"],
            graph.placeholders.audio_lengths: self.audio["ds_feats"],
            graph.placeholders.targets: self.targets["indices"],
            graph.placeholders.target_lengths: self.targets["lengths"],
        }

        return self.examples, self.attack


class Validation:
    """
    A Feed which will be passed into DeepSpeech only to validate the success of
    an attack. No targeting data is used.

    TODO: This is not used (we run validation against the deepspeech package)

    :param batch: a batch of (at least) audio examples and targeting data
    """

    def __init__(self, batch):
        self.audio = batch.audios
        self.targets = batch.targets
        self.examples = None

    def create_feeds(self, audio_ph, lens_ph):
        """
        Create the actual feeds for a validation test.

        :param audio_ph: the attack graph which holds the input placeholders
        :return: lens_ph for both plain examples and attack optimisation
        """
        # TODO - this is nasty!

        self.examples = {
            audio_ph: self.audio["padded_audio"],
            lens_ph: self.audio["ds_feats"]
        }

        return self.examples

