from py3nvml import py3nvml as pynvml


class GpuMemory:
    def __init__(self, gpu_device):

        pynvml.nvmlInit()
        self.__device_handle = pynvml.nvmlDeviceGetHandleByIndex(
            gpu_device
        )
        self.total = None
        self.used = None
        self.current_free = None
        self.previous_free = None
        self.max_batch = None
        self.all_batch = []

    def reset(self):
        self.used = None
        self.current_free = None
        self.previous_free = None
        self.max_batch = None
        self.all_batch = []

    def update_usage(self):

        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(
            self.__device_handle
        )

        self.total = gpu_memory.total
        self.used = gpu_memory.used

        if self.current_free is not None:
            self.previous_free = self.current_free

        self.current_free = gpu_memory.free

    def update_batch(self):
        self.all_batch.append(self.previous_free - self.current_free)
        self.max_batch = max(self.all_batch)

    def check_resource(self):
        return self.max_batch > self.current_free * 0.9

