import json
import numpy as np

from os import path
from cleverspeech.utils.Utils import dump_b64bytes


class SingleJsonDB:
    def __init__(self, directory, prefix="[\n", postfix="\n]"):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

        self.db = None

    def open(self, name):
        file_path = path.join(self.directory, name + ".json")
        self.db = open(file_path, mode='w+', encoding='utf-8')
        return self

    def _write(self, data):
        with self.db as f:
            f.write(self.prefix)
            json.dump(data, f, indent=2)
            f.write(self.postfix)

    def put(self, results_dict):

        data = {}

        for k in results_dict.keys():

            values = results_dict[k]

            if type(values) in [np.float32, np.int32, np.int16, np.int64]:
                values = float(values)

            if type(values) is np.ndarray:
                values = values.tolist()

            if type(values) is list:
                if type(values[0]) is np.ndarray:
                    for idx, value in enumerate(values):
                        values[idx] = value.tolist()

            if type(values) is not list:
                values = [values]

            data[k] = values

        self._write(data)

        return self


class FullJsonDB:
    def __init__(self, name="adv"):

        assert type(name) is str

        self.name = name
        self.current = None
        self.data = {}

    def open(self, schema_str):

        assert type(schema_str) is str
        self.current = self.name + "_" + schema_str + ".json"

        if path.exists(self.current) is False:
            with self._open_file(mode='w+') as f:
                f.write('[\n')

        else:
            with self._open_file(mode='w+') as f:
                f.write('\n')

        return self

    def _open_file(self, mode='a+'):
        return open(self.current, mode=mode, encoding='utf-8')

    def write(self):

        with self._open_file() as f:
            json.dump(self.data, f, indent=2)
            f.write(',\n')

        self.data = {}

        return self

    def close(self):

        with self._open_file() as f:
            f.write('{}\n]')

    def add(self, results_dict):

        for k in results_dict.keys():

            values = results_dict[k]

            if type(values) in [np.float32, np.int32, np.int16]:
                values = float(values)

            if type(values) is np.ndarray:
                values = dump_b64bytes(values)

            if type(values) is not list:
                values = [values]

            self.data[k] = values

        return self


def step_logging(step_results):
    s = ""
    for k, v in step_results.items():
        if type(v) in (float, int, np.float32, np.float64):
            s += "{k}: {v:.3f}\t".format(k=k, v=v)
        elif type(v) is str:
            s += "{k}: {v}\t".format(k=k, v=v)
        else:
            pass
    return s


def success_logging(example_result):

    s = """\rFound new example: {f}\tBound: {b:.3f}\tDistance: {d:.3f}\tLoss: {l:.3f}\tLoglike: {p:.3f}\tDecoding: {o}\tTarget: {t}\t""".format(
        f=example_result['basename'],
        b=example_result["bound"],
        d=example_result["distance"],
        l=example_result["total loss"],
        p=example_result["loglike"],
        o=example_result["decoding"],
        t=example_result["target"],
    )

    return s




