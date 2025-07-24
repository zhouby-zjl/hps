import time
import tracemalloc
from multiprocessing.managers import SyncManager
import numpy as np

import config


class tracer:
    enable_trace_time = True
    enable_trace_space = True
    enable_trace_iterations = True
    enable_trace_path_trace_delay = True
    enable_trace_retrying_times = True
    enable_trace_cpus = False
    time_records = {}
    space_records = {}
    iterations_records = {}
    path_trace_delay_records = []
    retrying_times_records = []

    @staticmethod
    def func_time(fn, func, path, *args):
        if not tracer.enable_trace_time:
            return func(*args)
        start_time_ns = time.time() * 1e9
        result = func(*args)
        execution_time_ns = int(time.time() * 1e9 - start_time_ns)
        path_sim_len = -1
        if path is not None:
            path_sim_len = (len(path) - 3) // 2
        tracer.time(fn, execution_time_ns, path_sim_len)
        return result

    @staticmethod
    def time(func_name : str, time_ns, path_sim_len, cpus_to_use=[]):
        if func_name == 'CompChangeParallel' and len(cpus_to_use) == 0:
            print("here")
        if func_name not in tracer.time_records:
            tracer.time_records[func_name] = [[time_ns, path_sim_len, cpus_to_use]]
        else:
            tracer.time_records[func_name].append([time_ns, path_sim_len, cpus_to_use])

    @staticmethod
    def space(func_name : str, space_B):
        if func_name not in tracer.space_records:
            tracer.space_records[func_name] = [space_B]
        else:
            tracer.space_records[func_name].append(space_B)

    @staticmethod
    def iterations(cpu_id, iters):
        if cpu_id not in tracer.iterations_records:
            tracer.iterations_records[cpu_id] = []
        tracer.iterations_records[cpu_id].append(iters)

    @staticmethod
    def path_trace(path_len, probe_delay):
        if tracer.enable_trace_path_trace_delay:
            tracer.path_trace_delay_records.append([path_len, probe_delay])

    @staticmethod
    def retrying_times(retrying_times):
        if tracer.enable_trace_retrying_times:
            tracer.retrying_times_records.append(retrying_times)

    @staticmethod
    def log(str):
        if not config.KEEP_SILENCE:
            print(str)

    @staticmethod
    def dump_time_csv():
        s = ""
        for fn in tracer.time_records:
            s += f"{fn}," + ",".join([
                str(x[0]) + '|' + str(x[1]) +
                ('|' + '#'.join(str(i) for i in x[2]) if tracer.enable_trace_cpus and len(x[2]) > 0 else '')
                for x in tracer.time_records[fn]
            ]) + "\n"
        return s

    @staticmethod
    def dump_space_csv():
        s = ""
        for fn in tracer.space_records:
            s += f"{fn},{','.join([str(x) for x in tracer.space_records[fn]])}\n"
        return s

    @staticmethod
    def dump_iterations_csv():
        s = ""
        for cpu_id in tracer.iterations_records:
            s += f"{cpu_id},{','.join([str(x) for x in tracer.iterations_records[cpu_id]])}\n"
        return s

    @staticmethod
    def dump_path_trace_csv():
        s = ""
        for rec in tracer.path_trace_delay_records:
            s += f"{rec[0]},{rec[1]}\n"
        return s

    @staticmethod
    def dump_retrying_times_csv():
        s = ""
        for rt in tracer.retrying_times_records:
            s += f"{rt}\n"
        return s

    @staticmethod
    def dump_all_to_files(file_path_prefix):
        s = tracer.dump_time_csv()
        tracer.write_to_file(s, file_path_prefix + 'time')
        s = tracer.dump_space_csv()
        tracer.write_to_file(s, file_path_prefix + 'space')
        s = tracer.dump_path_trace_csv()
        tracer.write_to_file(s, file_path_prefix + 'path-trace')
        s = tracer.dump_iterations_csv()
        tracer.write_to_file(s, file_path_prefix + 'iterations')
        s = tracer.dump_retrying_times_csv()
        tracer.write_to_file(s, file_path_prefix + 'retrying-times')

    @staticmethod
    def write_to_file(text, file_path):
        try:
            with open(file_path, 'w') as file:
                file.write(text)
            print(f"Dumped to {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def write_sim_stats(file_path_prefix, epr_cht, epr_sht, pss, 
                        n_endpoints, n_switches, n_flows,
                        jr_orig, jr_expected, jr_selected, net_desc):
        s = f"epr_cht,{','.join([str(x) for x in epr_cht])}\n" \
            f"epr_sht,{','.join([str(x) for x in epr_sht])}\n" \
            f"pss,{pss}\n" \
            f"n_endpoints,{n_endpoints}\n" \
            f"n_switches,{n_switches}\n" \
            f"n_flows,{n_flows}\n" \
            f"jr_orig,{jr_orig}\n" \
            f"jr_expected,{jr_expected}\n" \
            f"jr_selected,{jr_selected}\n"
        s += "\n".join([k + "," + str(net_desc[k]) for k in net_desc])
        tracer.write_to_file(s, file_path_prefix + 'stats')

    @staticmethod
    def clear():
        tracer.time_records = {}
        tracer.space_records = {}
        tracer.iterations_records = {}
        tracer.path_trace_delay_records = []
        tracer.cpus_used_records = []

class perf_worker:
    worker_time = None
    worker_space = None
    worker_compcri = None
    worker_searchdelta = None
    worker_iterations = None
    worker_time_map = None
    worker_space_map = None
    n_cpus = 0

    def __init__(self, manager : SyncManager, n_cpus):
        self.n_cpus = n_cpus
        self.worker_time = manager.list([0] * n_cpus)
        self.worker_compcri = manager.list([0] * n_cpus)
        self.worker_searchdelta = manager.list([0] * n_cpus)
        self.worker_iterations = manager.list([0] * n_cpus)
        self.worker_space = manager.list([0] * n_cpus)

        self.worker_time_map = {'worker': self.worker_time, 'CompCri': self.worker_compcri,
                           'SearchDelta': self.worker_searchdelta}
        self.worker_space_map = {'worker': self.worker_space}

    def copy_from_local_tracer_data(self, cpu_id, iterations=0):
        if tracer.enable_trace_time:
            for fn in self.worker_time_map:
                if fn in tracer.time_records and len(tracer.time_records[fn]) > 0:
                    w_time = tracer.time_records[fn]
                    if fn == 'worker':
                        self.worker_time_map[fn][cpu_id] = w_time[len(w_time) - 1]
                    else:
                        self.worker_time_map[fn][cpu_id] = int(np.round(np.mean(w_time)))

        if tracer.enable_trace_space:
            for fn in self.worker_space_map:
                if fn in tracer.space_records and len(tracer.space_records[fn]) > 0:
                    w_space = tracer.space_records[fn]
                    self.worker_space_map[fn][cpu_id] = w_space[len(w_space) - 1]

        if tracer.enable_trace_iterations:
            self.worker_iterations[cpu_id] = iterations

    def gather_statistics(self):
        if tracer.enable_trace_time:
            for fn in self.worker_time_map:
                wt = self.worker_time_map[fn]
                for i in range(self.n_cpus):
                    tracer.time(fn, wt[i], None)

        if tracer.enable_trace_space:
            for fn in self.worker_space_map:
                ws = self.worker_space_map[fn]
                for i in range(self.n_cpus):
                    tracer.space(fn, ws[i])

        if tracer.enable_trace_iterations:
            for cpu_id in range(self.n_cpus):
                tracer.iterations(cpu_id, self.worker_iterations[cpu_id])

def get_path_sim_len(fn, args):
    path_sim_len = -1
    if fn == 'RetrySelectPath':
        path_sim_len = (len(args[2]) - 2) // 2
    elif fn == 'SelectPath':
        path_sim_len = (len(args[2]) - 2) // 2
    elif fn == 'CompChange':
        path_sim_len = len(args[1])
    elif fn == 'CompCri':
        path_sim_len = len(args[2])

    return  path_sim_len

def trace_time(fn=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not tracer.enable_trace_time:
                return func(*args, **kwargs)
            func_name = fn if fn else func.__name__
            path_sim_len = get_path_sim_len(fn, args)
            start_time_ns = time.time() * 1e9
            result = func(*args, **kwargs)
            execution_time_ns = int(time.time() * 1e9 - start_time_ns)
            cpus_to_use = []
            if func_name == 'RetrySelectPath':
                cpus_to_use = result[2]
            elif func_name == 'SelectPath':
                cpus_to_use = result[1]
            elif func_name == 'CompChangeParallel':
                cpus_to_use = [result[1]]
            tracer.time(func_name, execution_time_ns, path_sim_len, cpus_to_use)
            return result
        return wrapper
    return decorator

def trace_space(fn=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not tracer.enable_trace_space:
                return func(*args, **kwargs)
            func_name = fn if fn else func.__name__
            tracemalloc.start()
            start_snapshot = tracemalloc.take_snapshot()
            result = func(*args, **kwargs)
            end_snapshot = tracemalloc.take_snapshot()
            stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            peak_memory_usage = 0
            for stat in stats:
                peak_memory_usage = max(peak_memory_usage, stat.size_diff)
            tracemalloc.stop()
            tracer.space(func_name, peak_memory_usage)
            return result
        return wrapper
    return decorator
