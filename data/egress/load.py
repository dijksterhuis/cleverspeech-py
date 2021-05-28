import bz2
import json
import boto3

import numpy as np

from os import path, makedirs
from abc import ABC, abstractmethod

from cleverspeech.utils import WavFile


def convert_types_for_json(results):
    data = {}

    for k in results.keys():

        values = results[k]

        if type(values) in [np.float32, np.int32, np.int16, np.int64]:
            values = float(values)

        if type(values) is np.ndarray:
            values = values.tolist()

        if type(values) is list:
            if type(values[0]) is np.ndarray:
                for idx, value in enumerate(values):
                    values[idx] = value.tolist()

            if type(values[0]) in [np.float32, np.int32, np.int16, np.int64]:
                values = [float(v) for v in values]

        if type(values) is not list:
            values = [values]

        data[k] = values

    return data


class Database(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def open(self, table):
        pass

    @abstractmethod
    def put(self, data):
        pass


class LocalWavFiles(Database):
    def __init__(self, db_location, bit_depth=2**15):
        self.db_location = db_location
        self.__table = None
        self.__bit_depth = bit_depth
        super().__init__()

    def open(self, table_name):

        if not path.exists(self.db_location):
            makedirs(self.db_location, exist_ok=True)

        self.__table = path.join(self.db_location, table_name + ".wav")

    def put(self, data):
        WavFile.write(
            self.__table,
            data,
            sample_rate=16000,
            bit_depth=16
        )


class LocalJsonMetadataFile(Database):
    def __init__(self, db_location, prefix="[\n", postfix="\n]"):
        self.db_location = db_location
        self.__prefix = prefix
        self.__postfix = postfix
        self.__table = None
        super().__init__()

    def open(self, table_name):

        if not path.exists(self.db_location):
            makedirs(self.db_location, exist_ok=True)

        file_path = path.join(self.db_location, table_name + ".json")
        self.__table = open(file_path, mode='w+', encoding='utf-8')

    def put(self, data):

        data = convert_types_for_json(data)

        with self.__table as f:
            f.write(self.__prefix)
            json.dump(data, f, indent=2)
            f.write(self.__postfix)


class S3JsonMetadataFile(Database):
    def __init__(self, db_location, prefix="[\n", postfix="\n]"):
        self.db_location = db_location  # the s3 bucket
        self.__prefix = prefix
        self.__postfix = postfix
        self.__table = None
        super().__init__()

    def open(self, table_name):

        file_path = table_name + ".json.bz2"
        file_path = file_path.lstrip("./")

        s3 = boto3.resource('s3')
        s3_bucket = s3.Bucket(self.db_location)
        self.__table = s3_bucket.Object(file_path)

    def put(self, data):

        data = convert_types_for_json(data)

        data = json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("ascii")

        # bz2 compression can reduce file size up to 4x, helpful when charged
        # for put requests... to decompress: bz2.decompress(compressed_data)

        compressed = bz2.compress(data)

        self.__table.put(
            Body=compressed,
            Tagging="costing:cleverSpeech"
        )

