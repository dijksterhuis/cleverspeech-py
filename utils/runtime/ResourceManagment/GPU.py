from py3nvml import py3nvml as pynvml
from cleverspeech.utils.Utils import l_map


class GpuMemory:
    def __init__(self, gpu_device, budget_factor=0.8):

        self.__nvml_init(gpu_device)

        self.budget_factor = budget_factor

        self.total = None
        self.used = None
        self.free = None

        self.total_processes = None
        self.max_processes = None
        self.all_processes = None

        self.all_batch = []
        self.pid_mem_usage = {}

    def __nvml_init(self, gpu_device):
        pynvml.nvmlInit()
        self.__device_handle = pynvml.nvmlDeviceGetHandleByIndex(
            gpu_device
        )

    def reset(self):
        self.used = None
        self.free = None

        self.total_processes = None
        self.max_processes = None
        self.all_processes = None

        self.all_batch = []
        self.pid_mem_usage = {}

    def update_usage(self):

        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(
            self.__device_handle
        )

        self.total = gpu_memory.total
        self.used = gpu_memory.used
        self.free = gpu_memory.free

    def update_pid_usage(self, processes):

        all_compute_processes = l_map(
            lambda x: [x.pid, x.usedGpuMemory],
            pynvml.nvmlDeviceGetComputeRunningProcesses(
                self.__device_handle
            )
        )

        our_pids = [p[0].pid for p in processes]

        self.pid_mem_usage = {}
        for pid, mem_usage in all_compute_processes:
            if pid in our_pids:
                self.pid_mem_usage[pid] = mem_usage

    def update_process_stats(self):
        self.all_processes = [m for m in self.pid_mem_usage.values()]
        self.total_processes = sum(self.all_processes)

        # max won't return 0 if list is empty. if there are no processes we
        # want to return some sort of bytes object, so just use the value from
        # the sum (they'll both be zero anyway).
        if len(self.all_processes) == 0:
            self.max_processes = self.total_processes

        else:
            self.max_processes = max(self.all_processes)

    def check_resource(self):
        return self.max_processes > self.free * self.budget_factor

