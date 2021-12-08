import os
from cleverspeech import data
from cleverspeech.utils.Utils import Logger


def create_batch_gen_fn(settings):

    data_loader = settings["data_loader"]

    if "mcv" in data_loader:
        settings["data_major"] = data_loader.split("-")[0]
        settings["data_minor"] = data_loader.split("-")[1]
        batch_gen = create_mcv_batch_gen_fn(settings)

    elif data_loader == "json":
        assert "audio_dir" in settings.keys()
        batch_gen = create_batch_gen_from_json_files(settings)

    elif data_loader == "csv":
        assert "audio_csv" in settings.keys()
        assert "targets_csv" in settings.keys()
        batch_gen = create_batch_gen_from_csv_files(settings)

    else:
        raise NotImplementedError

    return batch_gen


def create_mcv_batch_gen_fn(settings):

    data_major_id, data_minor_id = settings["data_major"], settings["data_minor"]

    if data_major_id == "mcv7":
        mcv_data = data.ingress.mcv7

    elif data_major_id == "mcv1":
        mcv_data = data.ingress.mcv1

    else:
        Logger.warn("You selected an incorrect dataset combination...")
        Logger.warn("Defaulting to mcv7-singlewords")
        data_major_id, data_minor_id = "mcv7", "sentences"
        mcv_data = data.ingress.mcv7

    base_dir = "./samples/{}/{}/".format(data_major_id, data_minor_id)

    audio_dir = os.path.join(base_dir, "all")
    transcript_file = "test.csv" if data_major_id == "mcv7" else "cv-valid-test.csv"
    transcript_file = os.path.join(base_dir, transcript_file)

    s3_tar_file_path = "{}-{}.tar.gz".format(data_major_id, data_minor_id)
    data.ingress.downloader.download(s3_tar_file_path)

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


def create_batch_gen_from_json_files(settings):

    audios = data.ingress.two_stage.TwoStageStandardAudioBatchETL(
        settings["audio_dir"],
        filter_term="audio.wav",
        file_size_sort="shuffle",
    )

    transcriptions = data.ingress.two_stage.TwoStageTranscriptions(
        settings["audio_dir"],
    )

    batch_gen = data.ingress.two_stage.TwoStageIterableBatches(
        settings, audios, transcriptions
    )

    return batch_gen


def create_batch_gen_from_csv_files(settings):

    audios = data.ingress.two_stage.TwoStageStandardAudioBatchETL(
        settings["audio_csv"],
        filter_term="audio.wav",
        file_size_sort="shuffle",
    )

    transcriptions = data.ingress.two_stage.TwoStageTranscriptions(
        settings["targets_csv"],
    )

    batch_gen = data.ingress.two_stage.TwoStageIterableBatches(
        settings, audios, transcriptions
    )

    return batch_gen