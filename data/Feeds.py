class Validation:
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

    def create_feeds(self, audio_ph, lens_ph):
        """
        Create the actual feeds

        :param graph: the attack graph which holds the input placeholders
        :return: feed_dict for both plain examples and attack optimisation
        """
        # TODO - this is nasty!

        self.examples = {
            audio_ph: self.audio["padded_audio"],
            lens_ph: self.audio["ds_feats"]
        }

        return self.examples


class Attack:
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
        Create the actual feeds

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

