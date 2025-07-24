import itertools
import numpy as np
from numpy import infty

from dcn_networks import sl_network, packet, endpoint, switch
import config
import random
from tracer import tracer, trace_time, trace_space, perf_worker
from rtt_simulator import rtt_simulator
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import math
import string
import struct
import mmap
import pickle
import os
from datetime import datetime
from collections import defaultdict

PRINT_FLAG = False

class controller:
    stop_flag_mmap = None
    stop_flag_mmap_f = None
    stop_flag_mmap_f_path = None

    class flow_info:
        idx = 0  # only used for NS-3
        node_src_id = 0
        node_dst_id = 0
        port_src = 0
        port_dst = 0
        event_time = 0 # only used for NS-3
        flow_size = 0  # only used for NS-3
        flow_size_normalized = 0   # only used for NS-3

        def __repr__(self):
            return f"({self.node_src_id}:{self.port_src} -> {self.node_dst_id}:{self.port_dst})"

    class InterruptException(Exception):
        pass

    net = None
    switch_ids = None
    n_nexthop_all = None
    deltas = None
    o_delta_comb_map_all = None
    o_delta_sing_map_all = None
    o_delta_tables_all = None
    switch_ids = None

    perf_worker_time = None
    perf_worker_space = None

    # batch_size is in the number of combinations
    def __init__(self, net : sl_network, n_cpus=1, batch_size=10):
        self.net = net
        self.switch_ids = self.net.get_switch_ids()
        self.o_delta_comb_map_all = None
        self.o_delta_tables_all = None
        self.hv_zero_map = {}

        self.update_hashes()
        self.n_cpus = n_cpus
        self.batch_size = batch_size
        if n_cpus >= 2:
            self.executor = ProcessPoolExecutor(max_workers=n_cpus)
            controller.mmap_create_stop_flag()
        else:
            self.executor = None

    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            controller.clean_mmap_stop_flag()

    @staticmethod
    def mmap_create_stop_flag():
        if controller.stop_flag_mmap is not None:
            return

        max_size = 4
        ascii_characters = string.ascii_letters + string.digits
        controller.stop_flag_mmap_f_path = "/tmp/" + ''.join(random.choice(ascii_characters) for _ in range(20))
        with open(controller.stop_flag_mmap_f_path, "wb") as controller.stop_flag_mmap_f:
            controller.stop_flag_mmap_f.write(b'\xff' * max_size)

        controller.stop_flag_mmap_f = open(controller.stop_flag_mmap_f_path, "r+b")
        controller.stop_flag_mmap = mmap.mmap(controller.stop_flag_mmap_f.fileno(), max_size)

    @staticmethod
    def mmap_open_stop_flag(f_path):
        if controller.stop_flag_mmap is None:
            max_size = 4
            controller.stop_flag_mmap_f = open(f_path, "r+b")
            controller.stop_flag_mmap = mmap.mmap(stop_flag_mmap_f.fileno(), max_size)
            controller.stop_flag_mmap_f_path = f_path

    @staticmethod
    def mmap_write_stop_flag(value):
        controller.stop_flag_mmap.seek(0)
        packed_value = struct.pack('I', value)
        controller.stop_flag_mmap.write(packed_value)
        controller.stop_flag_mmap.flush()

    @staticmethod
    def mmap_read_stop_flag():
        controller.stop_flag_mmap.seek(0)
        data = controller.stop_flag_mmap.read(4)
        value = struct.unpack('I', data)[0]
        return value

    @staticmethod
    def clean_mmap_stop_flag():
        if controller.stop_flag_mmap is not None:
            controller.stop_flag_mmap.close()
            controller.stop_flag_mmap = None

        if controller.stop_flag_mmap_f is not None:
            controller.stop_flag_mmap_f.close()
            controller.stop_flag_mmap_f = None

        if controller.stop_flag_mmap_f_path is not None:
            try:
                os.remove(controller.stop_flag_mmap_f_path)
            except Exception as e:
                print(f"Error removing the temporary file {controller.stop_flag_mmap_f_path}: {e}")
            controller.stop_flag_mmap_f_path = None

    def update_hashes(self):
        self.ecmp_compute_deltas()
        self.create_n_nexthop_all()
        self.compute_o_delta_tables_all()
        self.create_o_delta_comb_map_all()

    def create_n_nexthop_all(self):
        self.n_nexthop_all = {}
        for sw_id in self.switch_ids:
            sw = self.net.get_node_obj(sw_id)
            dsts = sw.routes.get_all_dsts()
            n_nexthop_set = set()
            for ip_dst in dsts:
                m = sw.routes.match_route(ip_dst)
                n_nexthop = len(m)
                n_nexthop_set.add(n_nexthop)

            n_nexthop_list = list(n_nexthop_set)
            n_nexthop_list.sort()
            self.n_nexthop_all[sw_id] = n_nexthop_list

    def ecmp_compute_deltas(self):
        self.deltas = []
        for i in range(config.HEADER_CHANGE_BITS):
            delta = 1 << i
            self.deltas.append(delta)

    def compute_o_delta_tables_all(self):
        self.o_delta_tables_all = {}
        for sw_id in self.n_nexthop_all:
            n_nexthop_list = self.n_nexthop_all[sw_id]
            sw = self.net.get_node_obj(sw_id)
            o_delta_table_sw = self.compute_o_delta_table(sw, n_nexthop_list)
            self.o_delta_tables_all[sw_id] = o_delta_table_sw

    def compute_o_delta_table(self, sw, n_nexthop_list):
        nhp_count = len(n_nexthop_list)
        o_delta_table = [[0] * nhp_count for _ in range(config.HEADER_CHANGE_BITS)]
        for j in range(nhp_count):
            n_nexthop = n_nexthop_list[j]
            hv_zero = sw.ecmp_hash_by_ba(packet.get_packet_zero(), n_nexthop)
            for i in range(config.HEADER_CHANGE_BITS):
                delta = self.deltas[i]
                hv_delta = sw.ecmp_hash_by_ba(packet.get_delta(delta), n_nexthop)
                o_delta = hv_delta ^ hv_zero
                o_delta_table[i][j] = o_delta

        return o_delta_table

    def evaluate_o_delta_comb(self):
        num_o_delta_cht_map = {}
        num_h_cht_map = {}
        num_o_delta_sht_map = {}
        for sw_id in self.o_delta_comb_map_all:
            o_delta_table = self.o_delta_comb_map_all[sw_id]
            for nnh in o_delta_table:
                if nnh == 1:
                    continue
                n_o_delta = len(o_delta_table[nnh].keys())
                if nnh not in num_o_delta_cht_map:
                    num_o_delta_cht_map[nnh] = []
                    num_h_cht_map[nnh] = []
                num_o_delta_cht_map[nnh].append(n_o_delta)
                num_h_cht_map[nnh] += [len(o_delta_table[nnh][k]) for k in o_delta_table[nnh]]

            sht = self.o_delta_sing_map_all[sw_id]
            for nnh in sht:
                if nnh == 1:
                    continue
                n_o_delta = len(sht[nnh])
                if nnh not in num_o_delta_sht_map:
                    num_o_delta_sht_map[nnh] = []

                num_o_delta_sht_map[nnh].append(n_o_delta)

        epr_cht = []
        for nnh in num_o_delta_cht_map:
            for n_o_delta in num_o_delta_cht_map[nnh]:
                epr_cht.append(n_o_delta / nnh)
        epr_sht = []
        for nnh in num_o_delta_sht_map:
            for n_o_delta in num_o_delta_sht_map[nnh]:
                epr_sht.append(n_o_delta / nnh)

        return epr_cht, epr_sht

    @trace_time('CreateTable')
    @trace_space('CreateTable')
    def create_o_delta_comb_map_all(self):
        self.o_delta_sing_map_all = {}
        self.o_delta_comb_map_all = {}
        for sw_id in self.o_delta_tables_all:
            o_delta_table = self.o_delta_tables_all[sw_id]
            o_delta_sing_sw, o_delta_comb_sw = self.create_o_delta_comb_map(sw_id, o_delta_table)
            self.o_delta_sing_map_all[sw_id] = o_delta_sing_sw
            self.o_delta_comb_map_all[sw_id] = o_delta_comb_sw

    def create_o_delta_comb_map(self, sw_id, o_delta_table):
        o_delta_comb_sw = {}
        o_delta_sing_sw = {}
        nnh_count = len(o_delta_table[0])

        for j in range(nnh_count):
            nnh = self.n_nexthop_all[sw_id][j]
            if nnh == 1:
                continue
            o_delta_sing = {}
            for i in range(config.HEADER_CHANGE_BITS):
                o_delta = o_delta_table[i][j]
                delta = self.deltas[i]
                if o_delta in o_delta_sing:
                    o_delta_sing[o_delta] |= delta
                else:
                    o_delta_sing[o_delta] = delta

            o_delta_sing_list = [[o_delta, o_delta_sing[o_delta]] for o_delta in o_delta_sing]
            o_delta_sing_list = sorted(o_delta_sing_list, key=lambda x: x[0])
            p = len(o_delta_sing_list)
            o_delta_comb_map = {}
            for h in range(1, np.power(2, p)):
                o_delta_comb = 0
                for k in range(p):
                    f = h >> k & 0x01
                    if f == 1:
                        o_delta_comb ^= o_delta_sing_list[k][0]

                if o_delta_comb in o_delta_comb_map:
                    o_delta_comb_map[o_delta_comb].append(h)
                else:
                    o_delta_comb_map[o_delta_comb] = [h]

            o_delta_sing_sw[nnh] = o_delta_sing_list
            o_delta_comb_sw[nnh] = o_delta_comb_map

        return o_delta_sing_sw, o_delta_comb_sw


    @trace_time('RetrySelectPath')
    @trace_space('RetrySelectPath')
    def designate_path_for_unknown_hash_algorithms_multiple_retrying(self, node_src_id, path, p: packet,
                                                                     max_retrying_times,
                                                                     only_use_sht=False,
                                                                     port_srcs_to_retry=[]):
        retried_port_srcs = set()
        port_src = 0
        port_src_max = 2**config.HEADER_CHANGE_BITS - 1
        port_src_min = 1
        avail_port_src = port_src_max - port_src_min + 1

        use_predetermined_port_srcs = True if len(port_srcs_to_retry) == max_retrying_times else False
        n_cpus_used_all_times = []
        for i in range(max_retrying_times):
            if len(retried_port_srcs) == avail_port_src:
                tracer.log("no available port src")
                return -1, i + 1, n_cpus_used_all_times

            if not use_predetermined_port_srcs:
                while True:
                    port_src = random.randint(port_src_min, port_src_max)
                    if port_src not in retried_port_srcs:
                        break
            else:
                port_src = port_srcs_to_retry[i]

            retried_port_srcs.add(port_src)
            p.port_src = port_src
            delta, n_cpus_used_all = self.designate_path_for_unknown_hash_algorithms(node_src_id, path, p,
                                                                    only_use_sht=only_use_sht)
            n_cpus_used_all_times += n_cpus_used_all

            if delta != -1:
                return port_src ^ delta, i + 1, n_cpus_used_all_times

        return -1, i + 1, n_cpus_used_all_times


    @trace_time('SelectPath')
    def designate_path_for_unknown_hash_algorithms(self, node_src_id, path, p: packet,
                                                   only_use_sht=False):
        path_ecmp = self.convert_path_ecmp(path, p, wo_hv_header=True)
        if path_ecmp == None:
            return -1

        deltas_sing_on_path = []

        delta_prev = 0
        flow = controller.flow_info()
        flow.node_src_id = node_src_id
        flow.node_dst_id = path[len(path) - 1]
        flow.port_src = p.port_src
        flow.port_dst = p.port_dst
        n_hops = 0

        path_ecmp_at_cur_hop = []

        n_cpus_used_all = []

        for node_id, nnh, ecmp_val in path_ecmp:
            n_hops += 1
            if n_hops == len(path):
                break
            if nnh == 1:
                continue

            # simulate UDP ping
            path_fd, ports_on_path, probe_delay = self.trace_ecmp_path(flow, ttl=n_hops + 1,
                                                                       src_port_delta=delta_prev,
                                                                       sim_delay=True)
            tracer.path_trace(len(path_fd), probe_delay)

            if len(path_fd) != n_hops + 2:
                tracer.log(f"designate_path_uha encounters an exception in acquiring an ecmp hashed port with "
                      f"ttl {n_hops}, delta {delta_prev}, path_fd {path_fd}, and path_expected {path}")
                return -1, n_cpus_used_all

            if path_fd[1 : n_hops + 1] != path[0 : n_hops]:
                tracer.log(f"designate_path_uha receives a different path_fd {path_fd} than path_expected {path}")
                return -1, n_cpus_used_all

            port_id = ports_on_path[n_hops]
            sw_cur = self.net.get_node_obj(path_fd[n_hops])
            m = sw_cur.routes.match_route(p.ip_dst)
            hv_header_w_delta = -1
            for idx in range(len(m)):
                if m[idx][0] == port_id:
                    hv_header_w_delta = idx
                    break

            if hv_header_w_delta == -1:
                tracer.log(f"designate_path_uha encounters an exception in computing hv_header")
                return -1, n_cpus_used_all

            o_delta = 0
            nnh_idx = self.n_nexthop_all[node_id].index(nnh)
            odt = self.o_delta_tables_all[node_id]

            for j in range(config.HEADER_CHANGE_BITS):
                if delta_prev >> j & 0x01 == 1:
                    o_delta ^= odt[j][nnh_idx]

            hv_header = hv_header_w_delta ^ o_delta

            path_ecmp_at_cur_hop += [(node_id, nnh, ecmp_val, hv_header)]
            o_delta_sing_list = self.o_delta_sing_map_all[node_id][nnh]
            deltas_sing_on_path.append(o_delta_sing_list)

            if path_fd[n_hops + 1] == path[n_hops]:
                continue

            if only_use_sht:
                delta = self.compute_valid_delta_recap(path_ecmp_at_cur_hop, deltas_sing_on_path)
                return delta, [1]

            if self.n_cpus >= 2:
                delta_prev, n_cpus_to_use = self.compute_valid_delta_parallel(path_ecmp_at_cur_hop, deltas_sing_on_path)
                n_cpus_used_all.append(n_cpus_to_use)

            else:
                delta_prev = self.compute_valid_delta_uha(path_ecmp_at_cur_hop, deltas_sing_on_path)

            if delta_prev == -1:
                tracer.log(f"designate_path_uha cannot find valid delta for n_hops {n_hops}")
                return -1, n_cpus_used_all

        return delta_prev, n_cpus_used_all

    def compute_valid_delta_recap(self, path_ecmp, deltas_sing_on_path):
        i = 0
        for node_id, nnh, ecmp_val, hv_header in path_ecmp:
            if nnh <= 1:
                continue
            o_delta_expected = ecmp_val ^ hv_header
            for item in deltas_sing_on_path[i]:
                if item[0] == o_delta_expected:
                    delta_bits = item[1]
                    for i in range(config.HEADER_CHANGE_BITS):
                        if delta_bits >> i & 0x01 == 1:
                            return 1 << i

            return -1
        return -1

    @trace_time('CompChange')
    def compute_valid_delta_uha(self, path_ecmp, deltas_sing_on_path):
        deltas_comb_on_path = []
        n_comb_on_path = []
        all_combs = 1
        n_entries = len(path_ecmp)

        for node_id, nnh, ecmp_val, hv_header in path_ecmp:
            o_delta_expected = ecmp_val ^ hv_header
            o_delta_comb_map = self.o_delta_comb_map_all[node_id][nnh]
            deltas_comb = o_delta_comb_map[o_delta_expected] if o_delta_expected in o_delta_comb_map else None
            deltas_comb_on_path.append(deltas_comb)

            n_deltas_comb = len(deltas_comb)
            n_comb_on_path.append(n_deltas_comb)
            all_combs *= n_deltas_comb

        for i in range(all_combs):
            rem = i
            o_comb_among_hops = [0] * n_entries
            for j in range(n_entries):
                o_comb_among_hops[j] = int(np.mod(rem, n_comb_on_path[j]))
                rem = (rem - o_comb_among_hops[j]) / n_comb_on_path[j]
                if rem == 0:
                    break

            criterias_all, zeros = controller.compute_criteria(deltas_sing_on_path, deltas_comb_on_path, o_comb_among_hops)
            a_valid_delta = controller.compute_valid_deltas(criterias_all, zeros, True)
            if a_valid_delta != -1:
                return a_valid_delta

        return -1

    @trace_time('CompChangeParallel')
    def compute_valid_delta_parallel(self, path_ecmp, deltas_sing_on_path):
        deltas_comb_on_path = []
        n_comb_on_path = []
        all_combs = 1

        for node_id, nnh, ecmp_val, hv_header in path_ecmp:
            o_delta_expected = ecmp_val ^ hv_header
            o_delta_comb_map = self.o_delta_comb_map_all[node_id][nnh]
            deltas_comb = o_delta_comb_map.get(o_delta_expected, [])
            deltas_comb_on_path.append(deltas_comb)
            n_comb_on_path.append(len(deltas_comb))
            all_combs *= len(deltas_comb)
        #print(f"all_combs: {all_combs}\n")
        if all_combs <= self.batch_size:
            r = controller.worker_batch(0, all_combs, deltas_sing_on_path, deltas_comb_on_path,
                                        n_comb_on_path, None)
            #print("one CPU")
            return r, 1

        next_start_i = 0
        futures = {}
        controller.mmap_write_stop_flag(0)
        #print(f"Mutiple CPUs: {self.n_cpus}\n")

        b = (all_combs + self.n_cpus - 1) // self.n_cpus

        for _ in range(self.n_cpus):
            end_i = min(next_start_i + b, all_combs)
            future = self.executor.submit(
                controller.worker_batch,
                next_start_i, end_i,
                deltas_sing_on_path,
                deltas_comb_on_path,
                n_comb_on_path,
                controller.stop_flag_mmap_f_path
            )
            futures[future] = (next_start_i, end_i)
            next_start_i = next_start_i + self.batch_size

        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for future in done:
            result = future.result()
            if PRINT_FLAG:
                print(f"---> [{datetime.now()}] result from {futures[future]}: {result}\n")

            if result != -1:
                controller.mmap_write_stop_flag(1)
                for f in futures:
                    f.cancel()
                    if PRINT_FLAG:
                        print(f"----> [{datetime.now()}] f.cancel() from {futures[f]}\n")
                return result, self.n_cpus

        return -1, self.n_cpus


    @staticmethod
    def worker_batch(start_i, end_i,
                     deltas_sing_on_path,
                     deltas_comb_on_path,
                     n_comb_on_path,
                     stop_flag_mmap_f):
        if stop_flag_mmap_f is not None:
            controller.mmap_open_stop_flag(stop_flag_mmap_f)

        start = time.time()
        n_entries = len(n_comb_on_path)
        for i in range(start_i, end_i):
            if stop_flag_mmap_f is not None and controller.mmap_read_stop_flag() == 1:
                if PRINT_FLAG:
                    print(f"[{datetime.now()}] Interrupted early in batch [{start_i}, {end_i})")
                return -1

            rem = i
            o_comb_among_hops = [0] * n_entries
            for j in range(n_entries):
                o_comb_among_hops[j] = int(np.mod(rem, n_comb_on_path[j]))
                rem = (rem - o_comb_among_hops[j]) / n_comb_on_path[j]
                if rem == 0:
                    break

            criterias_all, zeros = controller.compute_criteria(deltas_sing_on_path, deltas_comb_on_path, o_comb_among_hops)
            a_valid_delta = controller.compute_valid_deltas(criterias_all, zeros, True)
            if a_valid_delta != -1:
                if PRINT_FLAG:
                    print(
                        f"[{datetime.now()}] Batch time: {time.time() - start} from ({start_i}, {end_i}) with {a_valid_delta} to return")
                return a_valid_delta

        if PRINT_FLAG:
            print(f"[{datetime.now()}] Batch time: {time.time() - start} from ({start_i}, {end_i}) with -1 to return")
        return -1


    def closest_factors_with_limit(self, x, c):
        a = min(c, int(math.sqrt(x)))
        while a > 0:
            if x % a == 0:
                b = x // a
                return a, b
            a -= 1
        return 1, x

    #@trace_time('CompCri')
    @staticmethod
    def compute_criteria(deltas_sing_on_path, deltas_comb_on_path, o_comb_among_hops):
        criterias_all = []
        zeros = set()
        zeros.update(range(config.HEADER_CHANGE_BITS))
        n_entries = len(deltas_comb_on_path)
        for i in range(n_entries):
            comb = deltas_comb_on_path[i][o_comb_among_hops[i]]
            criterias = []
            for r in range(len(deltas_sing_on_path[i])):
                f = comb >> r & 0x01
                if f == 1:
                    delta = deltas_sing_on_path[i][r][1]
                    expected_o_delta = deltas_sing_on_path[i][r][0]
                    criterias.append([expected_o_delta, delta])
                    zeros_cur = set()
                    for t in range(config.HEADER_CHANGE_BITS):
                        f_delta = delta >> t & 0x01
                        if f_delta == 0:
                            zeros_cur.add(t)

                    zeros = zeros & zeros_cur
            criterias_all.append(criterias)

        return criterias_all, zeros

    def signal_handler(signum, frame):
        raise controller.InterruptException("interrupted by signal")

    #@trace_time('SearchDelta')
    @staticmethod
    def compute_valid_deltas(criterias_all, zeros, obtain_a_delta=False):
        bitmask = 0
        for pos in zeros:
            bitmask |= (1 << pos)

        v_range_l = -1
        v_range_r = -1
        v_prev = -1
        v_max_added = -1
        valid_deltas = []

        criterias_xored_all = []
        for criterias in criterias_all:
            criterias_xored = 0
            for cri in criterias:
                criterias_xored ^= cri[1]
            criterias_xored_all.append(criterias_xored)

        for v in range(2 ** config.HEADER_CHANGE_BITS):
            if ((~v & 0xffff) & bitmask) != bitmask:
                continue
            valid = True
            for j in range(len(criterias_all)):
                if criterias_xored_all[j] & v != v:
                    valid = False
                    break

                for cri in criterias_all[j]:
                    cri_fields = cri[1] & v
                    if cri_fields == 0:
                        valid = False
                        break
                    if cri[0] != 0 and np.mod(bin(cri_fields).count('1'), 2) == 0:
                        valid = False
                        break

                if not valid:
                    break

            if obtain_a_delta and valid:
                return v
            elif valid:
                if v_prev == -1:
                    v_range_l = v
                    v_prev = v
                    v_range_r = v
                elif v == v_prev + 1:
                    v_prev = v
                    v_range_r = v
                else:
                    valid_deltas.append((v_range_l, v_range_r))
                    v_max_added = v_range_r
                    v_range_l = v
                    v_prev = v
                    v_range_r = v

        if obtain_a_delta:
            return -1
        elif v_max_added < v_range_l:
            valid_deltas.append((v_range_l, v_range_r))

        return valid_deltas

    def designate_path_for_known_hash_algorithms(self, path, p : packet, obtain_a_delta=False):
        path_ecmp = self.convert_path_ecmp(path, p)
        if path_ecmp == None:
            return -1
        deltas_sing_on_path = []
        deltas_comb_on_path = []
        o_delta_expected_map = {}
        n_comb_on_path = []
        all_combs = 1
        i = 0
        for node_id, nnh, ecmp_val, hv_header in path_ecmp:
            o_delta_sing_list = self.o_delta_sing_map_all[node_id][nnh]
            o_delta_comb_map = self.o_delta_comb_map_all[node_id][nnh]
            o_delta_expected = ecmp_val ^ hv_header
            deltas_comb = o_delta_comb_map[o_delta_expected] if o_delta_expected in o_delta_comb_map else None
            deltas_sing_on_path.append(o_delta_sing_list)
            deltas_comb_on_path.append(deltas_comb)
            n_deltas_comb = len(deltas_comb)
            n_comb_on_path.append(n_deltas_comb)
            all_combs *= n_deltas_comb
            o_delta_expected_map[i] = o_delta_expected
            i+=1

        n_hops = len(deltas_comb_on_path)

        valid_deltas_all = []
        for i in range(all_combs):
            rem = i
            o_comb_among_hops = [0] * n_hops
            for j in range(n_hops):
                o_comb_among_hops[j] = int(np.mod(rem, n_comb_on_path[j]))
                rem = (rem - o_comb_among_hops[j]) / n_comb_on_path[j]
                if rem == 0:
                    break

            criterias_all = []
            zeros = set()
            zeros.update(range(config.HEADER_CHANGE_BITS))
            for j in range(n_hops):
                comb = deltas_comb_on_path[j][o_comb_among_hops[j]]
                criterias = []
                for r in range(len(deltas_sing_on_path[j])):
                    f = comb >> r & 0x01
                    if f == 1:
                        delta = deltas_sing_on_path[j][r][1]
                        expected_o_delta = deltas_sing_on_path[j][r][0]
                        criterias.append([expected_o_delta, delta])
                        zeros_cur = set()
                        for t in range(config.HEADER_CHANGE_BITS):
                            f_delta = delta >> t & 0x01
                            if f_delta == 0:
                                zeros_cur.add(t)

                        zeros = zeros & zeros_cur
                if len(criterias) == 1 and criterias[0] == 0xffff:
                    continue
                criterias_all.append(criterias)

            if obtain_a_delta:
                a_valid_delta = controller.compute_valid_deltas(criterias_all, zeros, True)
                if a_valid_delta != -1:
                    return a_valid_delta
            else:
                valid_deltas = controller.compute_valid_deltas(criterias_all, zeros, False)
                valid_deltas_all = self.union_of_ranges(valid_deltas_all, valid_deltas)

        #tracer.log(valid_deltas_all)
        if obtain_a_delta:
            return  -1
        else:
            return valid_deltas_all

    def run_path_tests(self, ep_src_idx, ep_dst_idx):
        ep_src : endpoint = self.net.endpoints[ep_src_idx]
        ep_dst: endpoint = self.net.endpoints[ep_dst_idx]
        path_fd = self.validate_delta(ep_src.node_id, 0, ep_dst.node_id, 8000)

    def trace_ecmp_path(self, flow : flow_info, ttl=255, src_port_delta=0, sim_delay=False):
        ep_src: endpoint = self.net.get_node_obj(flow.node_src_id)
        pkt = packet()
        addr = self.net.get_dst_addr(flow.node_src_id, flow.node_dst_id)
        pkt.ip_src = ep_src.ports.ports_info[0].addr
        pkt.ip_dst = addr
        pkt.port_src = flow.port_src ^ src_port_delta
        pkt.port_dst = flow.port_dst
        pkt.ttl = ttl
        path_fd = ep_src.send_packet(pkt)
        path_fd = [flow.node_src_id] + path_fd
        ports_on_path = []
        for i in range(len(path_fd) - 1):
            node_id_cur = path_fd[i]
            node_id_next = path_fd[i + 1]
            port_id = self.net.get_port_id(node_id_cur, node_id_next)
            ports_on_path.append(port_id)

        probe_delay = 0
        for p_len in range(1, len(path_fd) + 1):
            rtt = rtt_simulator.compute_rtt(p_len)
            probe_delay += rtt

        if sim_delay:
            time.sleep(probe_delay)

        return path_fd, ports_on_path, probe_delay


    def gen_orig_ecmp_paths(self, src_dst_flows):
        paths_info = []
        for flow in src_dst_flows:
            path_ecmp, ports_on_path, probe_delay = self.trace_ecmp_path(flow)
            paths_info.append([flow, path_ecmp, ports_on_path])
        return paths_info

    def compute_link_jointness_ratio(self, paths):
        path_links = []
        for path in paths:
            links = set()
            start = 1
            path_len = len(path)
            if path_len >= 5:
                end = int((path_len + 1) / 2)
            else:
                end = 1
            for i in range(start, end):
                link = (path[i], path[i + 1])
                links.add(link)
            path_links.append(links)

        shared_link_count = 0
        total_paths = len(paths)

        for i in range(total_paths):
            for j in range(total_paths):
                if i != j and not path_links[i].isdisjoint(path_links[j]):
                    shared_link_count += 1
                    break

        link_jointness_ratio = shared_link_count / total_paths

        return link_jointness_ratio

    def gen_max_disjoint_paths(self, src_dst_flows):
        paths_selected = []
        links_selected = set()
        for flow in src_dst_flows:
            paths = self.net.paths_all[flow.node_src_id][flow.node_dst_id]
            overlapped_min = 10000000
            path_max_disjoint = None
            for p_can in paths:
                overlapped = 0
                for i in range(1, len(p_can) - 2):
                    lh = self.net.pair_hash(p_can[i], p_can[i + 1])
                    if lh in links_selected:
                        overlapped += 1

                if overlapped == 0:
                    overlapped_min = 0
                    path_max_disjoint = p_can
                    break
                if overlapped < overlapped_min:
                    overlapped_min = overlapped
                    path_max_disjoint = p_can

            if path_max_disjoint != None:
                for i in range(1, len(path_max_disjoint) - 1):
                    lh = self.net.pair_hash(path_max_disjoint[i], path_max_disjoint[i + 1])
                    links_selected.add(lh)
                paths_selected.append([path_max_disjoint, len(paths)])

        disjoint_paths = {}
        for path_info in paths_selected:
            ports_on_path = []
            path = path_info[0]
            for i in range(len(path) - 1):
                node_id_cur = path[i]
                node_id_next = path[i + 1]
                port_id = self.net.get_port_id(node_id_cur, node_id_next)
                ports_on_path.append(port_id)

            if path[0] not in disjoint_paths:
                disjoint_paths[path[0]] = {}

            disjoint_paths[path[0]][path[len(path) - 1]] = [path, ports_on_path, path_info[1]]

        return disjoint_paths

    def gen_max_disjoint_paths_for_flows(self, src_dst_flows):
        max_flow_size = max(f.flow_size for f in src_dst_flows if f.flow_size > 0)
        for f in src_dst_flows:
            f.flow_size_normalized = f.flow_size / max_flow_size if max_flow_size > 0 else 0.0

        disjoint_paths = {}
        link_load = defaultdict(float)
        alpha = 2

        for f in sorted(src_dst_flows, key=lambda x: -x.flow_size_normalized):  # prioritize large flows
            paths = self.net.paths_all[f.node_src_id][f.node_dst_id]
            weighted_overlap_min = float('inf')
            best_path = None

            for p_can in paths:
                lh_p = []
                for i in range(0, len(p_can) - 1):
                    lh = self.net.pair_hash(p_can[i], p_can[i + 1])
                    lh_p.append(link_load[lh])

                lh_p.sort(reverse=True)
                ln_p_len = len(lh_p)
                weighted_overlap = np.sum([np.power(alpha, ln_p_len - i) * lh_p[i] for i in range(ln_p_len)])

                if weighted_overlap < weighted_overlap_min:
                    weighted_overlap_min = weighted_overlap
                    best_path = p_can

            if best_path:
                for i in range(len(best_path) - 1):
                    lh = self.net.pair_hash(best_path[i], best_path[i + 1])
                    link_load[lh] += f.flow_size_normalized

                ports_on_path = []
                for i in range(len(best_path) - 1):
                    node_id_cur = best_path[i]
                    node_id_next = best_path[i + 1]
                    port_id = self.net.get_port_id(node_id_cur, node_id_next)
                    ports_on_path.append(port_id)

                disjoint_paths[f.idx] = [best_path, ports_on_path, len(paths)]
            else:
                disjoint_paths[f.idx] = [None, [], 0]

        return disjoint_paths

    def assign_max_disjoint_paths_for_flows(self, src_dst_flows):
        L = {}
        P_assign = {}
        for f in sorted(src_dst_flows, key=lambda x: -x.flow_size):
            paths = self.net.paths_all[f.node_src_id][f.node_dst_id]
            L_sum_min = infty
            L_sum_min_idx = -1
            for idx, p_can in enumerate(paths):
                L_sum = 0
                for i in range(len(p_can) - 1):
                    u, v = p_can[i], p_can[i + 1]
                    L_sum += L[(u, v)] if (u, v) in L else 0
                if L_sum < L_sum_min:
                    L_sum_min = L_sum
                    L_sum_min_idx = idx
            if L_sum_min_idx != -1:
                for i in range(len(paths[L_sum_min_idx]) - 1):
                    u, v = paths[idx][i], paths[idx][i + 1]
                    if (u, v) in L:
                        L[(u, v)] += f.flow_size
                    else:
                        L[(u, v)] = f.flow_size
                best_path = paths[L_sum_min_idx]

                ports_on_path = []
                for i in range(len(best_path) - 1):
                    node_id_cur = best_path[i]
                    node_id_next = best_path[i + 1]
                    port_id = self.net.get_port_id(node_id_cur, node_id_next)
                    ports_on_path.append(port_id)

                P_assign[(f.idx)] = [best_path, ports_on_path, len(paths)]
            else:
                P_assign[(f.idx)] = [None, [], 0]

        return P_assign


    def gen_max_disjoint_paths_for_flows_bruce_force(self, src_dst_flows):
        max_flow_size = max(f.flow_size for f in src_dst_flows if f.flow_size > 0)
        for f in src_dst_flows:
            f.flow_size_normalized = f.flow_size / max_flow_size if max_flow_size > 0 else 0.0

        paths_len_vector = []
        n_combs = 1
        n_flows = len(src_dst_flows)
        for f in src_dst_flows:
            paths_len = len(self.net.paths_all[f.node_src_id][f.node_dst_id])
            paths_len_vector.append(paths_len)
            n_combs *= paths_len

        link_load_min_all = float("inf")
        comb_idxes_opt = None
        for i in range(n_combs):
            rem = i
            comb_idxes = [0] * n_flows
            for j in range(n_flows):
                comb_idxes[j] = int(np.mod(rem, paths_len_vector[j]))
                rem = (rem - comb_idxes[j]) / paths_len_vector[j]
                if rem == 0:
                    break

            link_load = defaultdict(float)
            j = 0
            for f in src_dst_flows:
                p = self.net.paths_all[f.node_src_id][f.node_dst_id][comb_idxes[j]]
                for i in range(len(p) - 1):
                    lh = self.net.pair_hash(p[i], p[i + 1])
                    link_load[lh] += f.flow_size_normalized
                j += 1

            link_load_min = np.min([link_load[k] for k in link_load])
            if link_load_min < link_load_min_all:
                link_load_min_all = link_load_min
                comb_idxes_opt = comb_idxes.copy()

        disjoint_paths = {}
        if comb_idxes_opt is not None:
            for f in src_dst_flows:
                paths = self.net.paths_all[f.node_src_id][f.node_dst_id]
                best_path = paths[comb_idxes_opt[j]]
                ports_on_path = []
                for i in range(len(best_path) - 1):
                    node_id_cur = best_path[i]
                    node_id_next = best_path[i + 1]
                    port_id = self.net.get_port_id(node_id_cur, node_id_next)
                    ports_on_path.append(port_id)

                disjoint_paths[f.idx] = [best_path, ports_on_path, len(paths)]

        return disjoint_paths

    def get_designated_path(self, flow_req : flow_info, max_disjoint_paths, flowMode=False):
        if flowMode:
            if flow_req.idx not in max_disjoint_paths:
                return None
            designated_path = max_disjoint_paths[flow_req.idx]
        else:
            if flow_req.node_src_id not in max_disjoint_paths or \
                    flow_req.node_dst_id not in max_disjoint_paths[flow_req.node_src_id]:
                tracer.log(f"[Node {flow_req.node_src_id}] has no designated path reaching to dst node {flow_req.node_dst_id}")
                return None
            designated_path = max_disjoint_paths[flow_req.node_src_id][flow_req.node_dst_id]

        return designated_path

    def find_ep_port_src_uha(self, flow_req : flow_info, max_disjoint_paths, max_retrying_times,
                             only_use_sht=False,
                             port_srcs_to_retry=[], flowMode=False):
        designated_path = self.get_designated_path(flow_req, max_disjoint_paths, flowMode=flowMode)
        if designated_path is None:
            return flow_req.port_src, False, 0

        if designated_path[2] == 1:
            return flow_req.port_src, True, 0

        ep_src : endpoint = self.net.get_node_obj(flow_req.node_src_id)
        ip_dst = self.net.get_dst_addr(flow_req.node_src_id, flow_req.node_dst_id)
        if ip_dst == None:
            tracer.log(f"[Node {flow_req.node_src_id}] error to get the address of dst node {flow_req.node_dst_id}")
            return flow_req.port_src, False, 0

        pkt = packet()
        pkt.ip_src = ep_src.ports.ports_info[0].addr
        pkt.ip_dst = ip_dst
        pkt.port_src = flow_req.port_src
        pkt.port_dst = flow_req.port_dst

        #a_valid_delta = self.designate_path_for_unknown_hash_algorithms(ep_src.node_id, designated_path[0][1:], pkt)
        #if a_valid_delta != -1:
        #    return flow_req.port_src ^ a_valid_delta, True
        #else:
        #    return flow_req.port_src, False

        changed_port_src, retrying_times, n_cpus_used_all_times = (
            self.designate_path_for_unknown_hash_algorithms_multiple_retrying(
                                                                ep_src.node_id,
                                                                designated_path[0][1:],
                                                                pkt, max_retrying_times,
                                                                only_use_sht,
                                                                port_srcs_to_retry))

        if changed_port_src != -1:
            return changed_port_src, True, retrying_times
        else:
            return flow_req.port_src, False, retrying_times


    def find_ep_port_src(self, flow_req : flow_info, max_disjoint_paths, obtain_a_delta=True):
        designated_path = self.get_designated_path(flow_req, max_disjoint_paths)
        if designated_path is None:
            return flow_req.port_src, False

        if designated_path[2] == 1:
            return flow_req.port_src, True

        ep_src : endpoint = self.net.get_node_obj(flow_req.node_src_id)
        ip_dst = self.net.get_dst_addr(flow_req.node_src_id, flow_req.node_dst_id)
        if ip_dst == None:
            tracer.log(f"[Node {flow_req.node_src_id}] error to get the address of dst node {flow_req.node_dst_id}")
            return flow_req.port_src, False

        pkt = packet()
        pkt.ip_src = ep_src.ports.ports_info[0].addr
        pkt.ip_dst = ip_dst
        pkt.port_src = flow_req.port_src
        pkt.port_dst = flow_req.port_dst

        if obtain_a_delta:
            a_valid_delta = self.designate_path_for_known_hash_algorithms(designated_path[0][1:], pkt, True)
            if a_valid_delta != -1:
                return flow_req.port_src ^ a_valid_delta, True
            else:
                return flow_req.port_src, False
        else:
            valid_deltas = self.designate_path_for_known_hash_algorithms(designated_path[0][1:], pkt, False)
            if len(valid_deltas) > 0:
                vd = valid_deltas[0][0]
                return flow_req.port_src ^ vd, True
            else:
                return flow_req.port_src, False


    def validate_designated_flow(self, flow_req : flow_info, port_src_arb, disjoint_paths, flowMode=False):
        designated_path = self.get_designated_path(flow_req, disjoint_paths, flowMode=flowMode)
        if designated_path is None:
            return False, None, None

        ep_src: endpoint = self.net.get_node_obj(flow_req.node_src_id)

        addr = self.net.get_dst_addr(flow_req.node_src_id, flow_req.node_dst_id)
        if addr == None:
            tracer.log(f"[Node {flow_req.node_src_id}] error to get the address of dst node {flow_req.node_dst_id}")
            return False, None, None

        pkt = packet()
        pkt.ip_src = ep_src.ports.ports_info[0].addr
        pkt.ip_dst = addr
        pkt.port_src = port_src_arb
        pkt.port_dst = flow_req.port_dst
        path_fd = ep_src.send_packet(pkt)
        path_fd = [flow_req.node_src_id] + path_fd

        tracer.log("path (expected): " + str(designated_path[0]))
        tracer.log("path (actual): " + str(path_fd))

        valid = designated_path[0] == path_fd
        return valid, path_fd, designated_path



    def merge_ranges(self, ranges):
        if not ranges:
            return []

        ranges.sort(key=lambda x: x[0])

        merged_ranges = [ranges[0]]
        for current in ranges[1:]:
            last = merged_ranges[-1]

            if current[0] <= last[1] + 1:
                merged_ranges[-1] = (last[0], max(last[1], current[1]))
            else:
                merged_ranges.append(current)

        return merged_ranges

    def union_of_ranges(self, A, B):
        combined_ranges = A + B
        return self.merge_ranges(combined_ranges)

    def gen_candidate_ecmp_values(self, bm_te, p):
        cands = []
        for t in range(p):
            c = []
            for r in range(p):
                b = bm_te[r] & (1 << r)
                if b != 0:
                    c.append(b)
            cands.append(c)

        xc = []
        for lst in cands:
            combinations = lst[:]
            for r in range(2, len(lst) + 1):
                for combo in itertools.combinations(lst, r):
                    xor_result = 0
                    for num in combo:
                        xor_result ^= num
                    combinations.append(xor_result)
            xc.append(combinations)

        index_combinations = itertools.product(*[range(len(lst)) for lst in xc])

        Y = []
        for indices in index_combinations:
            xor_result = 0
            for i, idx in enumerate(indices):
                xor_result ^= xc[i][idx]
            Y.append(xor_result)

        return Y

    def convert_path_ecmp(self, path, p : packet, wo_hv_header=False):
        path_ecmp = []
        for i in range(len(path) - 1):
            node_id_cur = path[i]
            node_id_next = path[i + 1]
            port_id = self.net.get_port_id(node_id_cur, node_id_next)
            node_cur = self.net.get_node_obj(node_id_cur)
            routes = node_cur.routes.match_route(p.ip_dst)
            nnh = len(routes)
            if nnh == 0:
                return None
            hv_header = 0
            if not wo_hv_header and isinstance(node_cur, switch):
                hv_header = node_cur.ecmp_hash_by_ba(p.to_seq(), nnh)

            port_idx_in_ecmp = 0
            for route in routes:
                if route[0] == port_id:
                    break
                port_idx_in_ecmp += 1
            if port_idx_in_ecmp == nnh:
                return None

            if wo_hv_header:
                path_ecmp.append([node_id_cur, nnh, port_idx_in_ecmp])
            else:
                path_ecmp.append([node_id_cur, nnh, port_idx_in_ecmp, hv_header])
        return path_ecmp


    def save_hashmap_all(self):
        pass

    def load_hashmap_all(self):
        pass