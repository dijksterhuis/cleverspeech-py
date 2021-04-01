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
        while True:

            data = queue.get()

            if data == "dead":

                # This is a signal telling us there's no more data inbound.
                # Obviously we don't want to write this to disk, so tell parent
                # that we're done with this item and break the while loop.

                queue.task_done()
                break

            elif data != "dead":

                # There is still valid data in the Queue. Keep writing.
                # Make sure to tell the JoinableQueue when we're done processing
                # so it can block and unblock when the spawner context manager
                # exits.

                batched_outs = data

                for log_result, db_output in self.extracter.gen(batched_outs):

                    log(
                        log_result,
                        wrap=False,
                        outdir=self.outdir,
                        stdout=False,
                        timings=True,
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

                queue.task_done()


class FullFileWriter:
    def __init__(self, outdir, extracter):

        self.outdir = outdir
        self.extracter = extracter
        self.example_db = SingleJsonDB(outdir)

    def write(self, queue):
        while queue:
            if queue.empty() is not True:

                batched_outs = queue.get()

                log_result, db_output = self.extracter.gen(batched_outs)
                log(
                    log_result,
                    wrap=False,
                    outdir=self.outdir,
                    stdout=False,
                    timings=True,
                )

                if db_output is not None:

                    example_dir = db_output['basename']

                    if not path.exists(path.join(self.outdir, example_dir)):
                        mkdir(path.join(self.outdir, example_dir))

                    db_file_path = path.join(
                        example_dir, "/" + "step{}".format(db_output["step"])
                    )
                    self.example_db.open(db_file_path).put(db_output)

                    # -- Write audio data.
                    for wav_file in ["audio", "deltas", "advs"]:

                        outpath = path.join(self.outdir, db_file_path)
                        outpath += "_{}.wav".format(wav_file)

                        WavFile.write(
                            outpath,
                            db_output[wav_file],
                            sample_rate=16000,
                            bit_depth=16,
                        )


