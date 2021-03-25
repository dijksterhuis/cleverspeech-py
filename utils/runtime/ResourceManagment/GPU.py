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
        self.total_processes = sum(self.pid_mem_usage.values())
        self.max_processes = max(self.pid_mem_usage.values())
        self.all_processes = [m for m in self.pid_mem_usage.values()]

    def check_resource(self):
        return self.max_processes > self.free * self.budget_factor

