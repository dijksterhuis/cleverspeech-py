from os import path, mkdir

from cleverspeech.utils.Utils import log
from cleverspeech.data.egress.Databases import SingleJsonDB, FullJsonDB
from cleverspeech.utils import WavFile


class SingleFileWriter:
    def __init__(self, outdir, extracter):

        self.outdir = outdir
        self.extracter = extracter
        self.example_db = SingleJsonDB(outdir)

    def write(self, queue):
        while queue:
            if queue.empty() is not True:
                batched_outs = queue.get()

                for log_result, db_output in self.extracter.gen(batched_outs):

                    log(
                        log_result,
                        wrap=False,
                        outdir=self.outdir,
                        stdout=False
                    )

                    if db_output is not None:

                        db_file_path = db_output['basenames'].rstrip(".wav")

                        self.example_db.open(db_file_path).put(db_output)

                        # -- Write audio data.
                        for wav_file in ["audio", "deltas", "advs"]:

                            outpath = path.join(self.outdir, db_file_path)
                            outpath += "_{}.wav".format(wav_file)

                            WavFile.write(
                                outpath,
                                db_output[wav_file],
                                sample_rate=16000,
                                bit_depth=16
                            )


class FullFileWriter:
    def __init__(self, outdir, extracter):

        self.outdir = outdir
        self.extracter = extracter
        self.example_db = FullJsonDB(outdir)

    def write(self, queue):
        while queue:
            if queue.empty() is not True:

                batched_outs = queue.get()

                log_result, db_output = self.extracter.gen(batched_outs)
                log(log_result, wrap=False, outdir=self.outdir, stdout=False)

                if db_output is not None:

                    example_dir = path.join(self.outdir, db_output['basename'])

                    if not path.exists(example_dir):
                        mkdir(example_dir)

                    db_file_path = example_dir + "/" + "step{}".format(db_output["step"])
                    self.example_db.open(db_file_path).put(db_output)

                    # -- Write audio data.
                    for wav_file in ["audio", "deltas", "advs"]:

                        outpath = path.join(self.outdir, db_file_path)
                        outpath += "_{}.wav".format(wav_file)

                        WavFile.write(
                            outpath,
                            db_output[wav_file],
                            sample_rate=16000,
                            bit_depth=16
                        )


