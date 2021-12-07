import os
from cleverspeech.data.ingress import downloader
from cleverspeech.utils.Utils import Logger


def create_batch_gen_fn(settings):

    data_major_id, data_minor_id = settings["data_major"], settings["data_minor"]

    if data_major_id == "mcv7":
        from cleverspeech.data.ingress import mcv7 as mcv_data

    elif data_major_id == "mcv1":
        from cleverspeech.data.ingress import mcv1 as mcv_data

    else:
        Logger.warn("You selected an incorrect dataset combination...")
        Logger.warn("Defaulting to mcv7-sentences")
        data_major_id, data_minor_id = "mcv7", "sentences"
        from cleverspeech.data.ingress import mcv7 as mcv_data

    base_dir = "./samples/{}/{}/".format(data_major_id, data_minor_id)
    audio_dir = os.path.join(base_dir, "all")
    transcript_file = "test.csv" if data_major_id == "mcv7" else "cv-valid-test.csv"
    transcript_file = os.path.join(base_dir, transcript_file)

    s3_tar_file_path = "{}-{}.tar.gz".format(data_major_id, data_minor_id)
    downloader.download(s3_tar_file_path)

    audios = mcv_data.StandardAudioBatchETL(
        s3_tar_file_path,
        audio_dir,
        settings["max_examples"],
        filter_term=".wav",
        max_file_size=settings["max_audio_file_bytes"],
        file_size_sort="shuffle"
    )

    transcriptions = mcv_data.TranscriptionsFromCSVFile(
        transcript_file,
        settings["max_targets"],
    )

    batch_gen = mcv_data.IterableBatches(
        settings, audios, transcriptions
    )

    return batch_gen

