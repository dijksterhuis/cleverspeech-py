class AudiosBatch(object):
    def __init__(self, data, max_len):
        self.max_length = max_len
        [
            self.audio,
            self.basenames,
            self.padded_audio,
            self.actual_lengths,
            self.feature_lengths,
            self.alignment_lengths
        ] = data


class TargetsBatch(object):
    def __init__(self, data, tokens):
        self.tokens = tokens
        [
            self.phrases,
            self.ids,
            self.indices,
            self.lengths
        ] = data


class Batch:
    def __init__(self, size, audios, targets, feed_cls):

        self.size = size

        self.audios = audios

        self.targets = targets

        self.feeds = feed_cls(audios, targets)

    def create_feeds(self, *args, **kwargs):
        self.feeds.create_feeds(*args, **kwargs)


