import os
import boto3
import tarfile
# import multiprocessing as mp
from cleverspeech.utils.Utils import Logger
from progressbar import ProgressBar


def download(s3_archive):
    # tarfile_path = _create_tarfile_filepath(data_major_id, data_minor_id)

    samples_path = os.path.join(
        os.path.join("./samples", s3_archive.split("-")[0]),
        s3_archive.split("-")[-1].rstrip(".tar.gz")
    )

    if not os.path.exists(samples_path):
        # os.makedirs(os.path.join(samples_path, "all"), exist_ok=True)
        if not os.path.exists(s3_archive):
            _download_from_s3(s3_archive)

        _extract_from_tarfile(s3_archive)

    return True


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
                    bar.update(idx)
                    yield tarinfo

            tar.extractall(members=bar_updater(tar))


def _select_n_files(directory, n):
    pass
