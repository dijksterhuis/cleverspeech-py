import os
import sys
import shutil
import boto3
import tarfile
# import multiprocessing as mp
from cleverspeech.utils.Utils import Logger
from progressbar import ProgressBar
from botocore.exceptions import ClientError


def download(s3_archive):
    # tarfile_path = _create_tarfile_filepath(data_major_id, data_minor_id)

    samples_path = os.path.join(
        os.path.join("./samples", s3_archive.split("-")[0]),
        s3_archive.split("-")[-1].rstrip(".tar.gz")
    )
    audios_dir = os.path.join(samples_path, "all")

    try:
        assert os.path.exists(samples_path)        # base dir path exists

    except AssertionError:
        Logger.warn("{} directory doesn't exist locally.".format(samples_path))
        Logger.info("Downloading archive file {}.".format(s3_archive))

        _maybe_download_and_extract(s3_archive)

    else:

        try:
            more_than_one = len(os.listdir(os.path.join(samples_path, "all"))) > 2
            assert more_than_one  # at least 1 wav and json file are present
            assert len(os.listdir(samples_path)) == 2  # transcription csv
            assert os.path.exists(audios_dir)          # audios dir exists

            if more_than_one:
                wav_count, json_count = 0, 0
                for fp in os.listdir(audios_dir):
                    json_count += 1 if ".json" in fp else 0
                    wav_count += 1 if ".wav" in fp else 0
                assert json_count > 0
                assert wav_count > 0
                assert wav_count == json_count

            else:
                raise AssertionError

        except AssertionError:
            Logger.warn("Incomplete sample data.")
            Logger.info(
                "Deleting and recreating {} from {}".format(samples_path, s3_archive)
            )
            shutil.rmtree(samples_path)
            os.makedirs(audios_dir, exist_ok=True)
            _maybe_download_and_extract(s3_archive)

    return True


def _maybe_download_and_extract(s3_archive):
    if not os.path.exists(s3_archive):
        try:
            _download_from_s3(s3_archive)
        except ClientError:
            Logger.critical(
                "Could not download {} from s3://cleverspeech-data".format(s3_archive)
            )
            Logger.critical("Does this archive file exist...?")
            sys.exit(os.EX_DATAERR)
    _extract_from_tarfile(s3_archive)


def _create_tarfile_filepath(data_major_id, data_minor_id):
    return "{}-{}.tar.gz".format(data_major_id, data_minor_id)


def _download_from_s3(tarfile_path, s3_bucket_name="cleverspeech-data"):
    s3 = boto3.client('s3')

    total_size = s3.head_object(Bucket=s3_bucket_name, Key=tarfile_path)["ContentLength"]

    Logger.info(
        "Downloading {} from S3 bucket {} ...".format(tarfile_path, s3_bucket_name)
    )

    with ProgressBar(max_value=total_size) as bar:

        def _upload_progress(chunk):
            bar.update(bar.value + chunk)

        s3.download_file(
            Bucket=s3_bucket_name,
            Key=tarfile_path,
            Filename=os.path.join("./", tarfile_path),
            Callback=_upload_progress,
        )

    Logger.info(
        "... Downloaded {}".format(tarfile_path)
    )


def _extract_from_tarfile(filepath):
    Logger.info("Extracting from archive: {}".format(filepath))

    with tarfile.open(filepath, "r:gz") as tar:
        with ProgressBar(max_value=len(tar.getmembers())) as bar:

            def bar_updater(members):
                for idx, tarinfo in enumerate(members):
                    if not os.path.isdir(tarinfo.path):
                        bar.update(idx)
                        yield tarinfo

            tar.extractall(members=bar_updater(tar))


def _select_n_files(directory, n):
    pass

