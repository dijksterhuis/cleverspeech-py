import tensorflow as tf
import tfdeterminism


class OOMEnabledSession(tf.Session):
    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):

        # note that oom reports can slow down the sess.run calls, so only use it
        # when something bad has definitely happened... see:
        # https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/core/protobuf/config.proto#L718

        try:
            return super().run(
                fetches,
                feed_dict=feed_dict,
                options=options,
                run_metadata=run_metadata
            )

        except tf.errors.ResourceExhaustedError:
            return super().run(
                fetches,
                feed_dict=feed_dict,
                options=tf.RunOptions(
                    report_tensor_allocations_upon_oom=True
                ),
                run_metadata=run_metadata
            )


class TFRuntime:
    def __init__(self, settings):

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        c = tf.ConfigProto()

        c.gpu_options.allow_growth = True
        c.allow_soft_placement = True

        if settings["enable_jit"] is True:
            jit_option = tf.OptimizerOptions.ON_1
            c.graph_options.optimizer_options.global_jit_level = jit_option

        self.config = c

        device = "/device:GPU:{}".format(settings["gpu_device"])

        self.session = OOMEnabledSession(config=self.config)

        self.device = tf.device(device)

        if settings["use_resource_variables"] is True:
            tf.enable_resource_variables()

        tf.reset_default_graph()

        # set the graph seed after resetting the graph...
        # https://stackoverflow.com/a/36289575/5945794

        tf.set_random_seed(settings["random_seed"])

        # noinspection PyBroadException
        try:
            # enable 1.15.5 deterministic GPU ops
            tfdeterminism.patch()
        except BaseException as e:
            # nvidia-tensorflow throws an error as it's patched internally
            # so just catch the error and move on
            pass

    @staticmethod
    def log_attack_tensors():
        tensors = []
        for op in tf.get_default_graph().get_operations():
            if "qq" in op.name:
                for tensor in op.values():
                    tensors.append(tensor.__str__())
        return "\n".join(tensors)
