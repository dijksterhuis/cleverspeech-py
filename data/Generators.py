from cleverspeech.data.Batches import Batch
from cleverspeech.utils.Utils import l_map, assert_positive_int


class BaseBatchGenerator:
    def __init__(self, size, total_numb):
        self.size = size
        self.numb_examples = total_numb

        self.batch = None
        self.id = None

    def popper(self, data):
        return l_map(
            lambda x: data.pop(x-1), range(self.size, 0, -1)
        )


def pop_target_phrase(all_targets, min_feats):
    candidate_target = all_targets.pop(0)
    if len(candidate_target[0]) > min_feats:
        pop_target_phrase(all_targets, min_feats)
    else:
        return candidate_target


class BatchGenerator(BaseBatchGenerator):
    def __init__(self, all_audio_file_paths, all_targets, batch_size):

        numb_examples = len(all_audio_file_paths)
        numb_targets = len(all_targets)

        assert_positive_int(batch_size)
        assert_positive_int(numb_examples)
        assert_positive_int(numb_targets)

        assert numb_targets * batch_size >= numb_examples

        self.all_audio_file_paths = all_audio_file_paths
        self.all_targets = all_targets

        super().__init__(batch_size, numb_examples)

    def generate(self, audio_etl_cls, target_etl_cls, feeds_cls):

        for idx in range(0, self.numb_examples, self.size):

            # Handle remainders: number of examples // desired batch size != 0
            if len(self.all_audio_file_paths) < self.size:
                self.size = len(self.all_audio_file_paths)

            # get n files paths and create the audio batch data
            audio_batch_data = self.popper(self.all_audio_file_paths)
            audios = audio_etl_cls(audio_batch_data)
            audio_batch = audios.extract().transform().load()

            #  we need to make sure target phrase length < n audio feats.
            # also, the first batch should also have the longest target phrase
            # and longest audio examples so we can easily manage GPU Memory
            # resources with the AttackSpawner context manager.
            target_phrase = pop_target_phrase(
                self.all_targets, min(audio_batch.alignment_lengths)
            )

            # each target must be the same length else numpy throws a hissyfit
            # because it can't understand skewed matrices
            target_batch_data = l_map(
                lambda _: target_phrase, range(self.size)
            )
            # actually load the n phrases as a batch of target data
            targets = target_etl_cls(target_batch_data)
            target_batch = targets.extract().transform().load()

            self.batch = Batch(
                self.size,
                audio_batch,
                target_batch,
                feeds_cls
            )

            self.id = idx

            yield self.id, self.batch


