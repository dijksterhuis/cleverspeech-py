import tensorflow as tf


class TFRuntime:
    def __init__(self, device_id=None):

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True

        if not device_id:
            device = "/device:GPU:0"
        else:
            device = "/device:GPU:{}".format(device_id)

        self.session = tf.Session(config=self.config)

        self.device = tf.device(device)

        tf.reset_default_graph()

    @staticmethod
    def log_attack_tensors():
        tensors = []
        for op in tf.get_default_graph().get_operations():
            if "qq" in op.name:
                for tensor in op.values():
                    tensors.append(tensor.__str__())
        return "\n".join(tensors)
