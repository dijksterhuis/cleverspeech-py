from os import path, mkdir

from cleverspeech.utils.Utils import dump_wavs, log
from cleverspeech.data.egress.Databases import SingleJsonDB, FullJsonDB


class SingleFileWriter:
    def __init__(self, outdir, extracter):

        self.outdir = outdir
        self.extracter = extracter
        self.example_db = SingleJsonDB(outdir)

    def write(self, queue):
        while queue:
            if queue.empty() is not True:
                batched_outs = queue.get()

                log_result, db_output = self.extracter.run(batched_outs)
                log(log_result, wrap=False, outdir=self.outdir, stdout=False)

                if db_output is not None:

                    db_file_path = db_output['basenames'].rstrip(".wav")

                    self.example_db.open(db_file_path).put(db_output)

                    # -- Write audio data.
                    dump_wavs(
                        self.outdir,
                        db_output,
                        ["audio", "deltas", "advs"],
                        filepath_key="basenames",
                        sample_rate=16000
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

                log_result, db_output = self.extracter.run(batched_outs)
                log(log_result, wrap=False, outdir=self.outdir, stdout=False)

                if db_output is not None:

                    example_dir = path.join(self.outdir, db_output['basename'])

                    if not path.exists(example_dir):
                        mkdir(example_dir)

                    db_file_path = example_dir + "/" + "step{}".format(db_output["step"])
                    self.example_db.open(db_file_path).put(db_output)

                    # -- Write audio data.
                    dump_wavs(
                        example_dir,
                        db_output,
                        ["audios", "deltas", "advs"],
                        filepath_key="basenames",
                        sample_rate=16000
                    )

