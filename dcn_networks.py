import copy

from rib import port_table, route_table
import numpy as np
import zlib as zl
from collections import deque
import networkx as nx
import socket
import matplotlib.pyplot as plt
import random
from abc import abstractmethod
import multiprocessing
from tqdm import tqdm
import threading
from sympy import factorint
from tracer import tracer
from enum import Enum
from path_generator_with_ecmp_hash_linearity import path_generator_with_ecmp_hash_linearity
import copy
from collections import defaultdict
import csv
from pathlib import Path

pkt_header_size = 12


class dcn_network:
    G = None
    G_original = None
    n = 0
    n_ep = 0
    n_links = 0
    endpoints = None
    map_from_edge_to_port_addr = None
    paths_all = None
    next_node_id = None
    ip_addresses = None
    target_simai = False
    target_simai_host_id_count = 0
    target_simai_switches = None
    target_simai_hosts = None

    @abstractmethod
    def __init__(self):
        self.n_links = 0
        self.target_simai = False
        self.target_simai_switches = None
        self.target_simai_hosts = None

    def build_topo(self):
        self.generate_ip_addresses()
        self.create_topology()
        tracer.log("Topology has been created")

    def build_topo_for_simai(self):
        self.target_simai = True
        self.generate_ip_addresses_for_simai()
        self.ip_allocation_seq = 0
        self.ip_allocation_idx_ip_to_seq_map = {}
        self.create_topology()
        tracer.log("Topology for SimAI has been created")

    def generate_ip_addresses_for_simai(self):
        self.ip_addresses = []
        for id in range(self.n_links + self.n_ep):  # plus the additional n_ep IP addresses for NVSwitch IDs
            ip_int = 0x0b000001 + ((id // 256) << 16) + ((id % 256) << 8)
            octets = [(ip_int >> (8 * i)) & 0xFF for i in range(4)][::-1]
            ip_str = '.'.join(map(str, octets))
            self.ip_addresses.append(ip_str)

    def get_ip_addresses(self, idx, is_host=False, host_id=0, sw_id=0):
        if self.target_simai:
            if is_host:
                if self.target_simai_hosts is None:
                    self.target_simai_hosts = self.get_ep_ids()
                return self.ip_addresses[self.target_simai_hosts.index(host_id)]
            else:
                if self.target_simai_switches is None:
                    self.target_simai_switches = self.get_switch_ids()
                if idx not in self.ip_allocation_idx_ip_to_seq_map:
                    self.ip_allocation_idx_ip_to_seq_map[idx] = self.ip_allocation_seq
                    seq = self.ip_allocation_seq
                    self.ip_allocation_seq += 1
                else:
                    seq = self.ip_allocation_idx_ip_to_seq_map[idx]
                return self.ip_addresses[self.n_ep + seq]
        else:
            return self.ip_addresses[idx]

    def load_id_mapping_for_simai(self, log_prefix):
        id_mapping_path = log_prefix + 'simai-topo-id-map'
        self.id_map, self.id_rmap = self.read_id_mapping(id_mapping_path)

    def adjust_port_ids_to_align_with_simai(self, log_prefix):
        ports_conf_all = self.read_ports_from_log_dir(log_prefix)
        if ports_conf_all is None:
            return
        for [src_id, dst_id, src_intf_id, dst_intf_id] in ports_conf_all:
            src = self.get_node_obj(src_id)
            h_src = self.pair_hash(src_id, dst_id)
            addr_src = self.map_from_edge_to_port_addr[h_src]
            for port in src.ports.ports_info:
                if port.addr == addr_src:
                    if port.id != src_intf_id:
                        port.id = src_intf_id - 1

            dst = self.get_node_obj(dst_id)
            h_dst = self.pair_hash(dst_id, src_id)
            addr_dst = self.map_from_edge_to_port_addr[h_dst]
            for port in dst.ports.ports_info:
                if port.addr == addr_dst:
                    if port.id != dst_intf_id:
                        port.id = dst_intf_id - 1

        print("switch port IDs have been adjusted according to SimAI's configuration")

    def read_id_mapping(self, file_path_mapping):
        id_map = {}
        id_rmap = {}

        with open(file_path_mapping, "r", newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not row:
                    continue  # 跳过空行
                original_id = int(row[0])
                simai_id = int(row[1])
                id_map[original_id] = simai_id
                id_rmap[simai_id] = original_id

        return id_map, id_rmap

    def read_ports_from_log_dir(self, log_prefix):
        if self.id_map is None or self.id_rmap is None:
            print("please call load_id_mapping_for_simai first")
            return None

        ports_csv_path = log_prefix + 'simai-ports'
        ports_conf_all = []

        with open(ports_csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                src_id, dst_id, src_intf_id, dst_intf_id = map(int, row)
                src_id_mapped = self.id_rmap[src_id]
                dst_id_mapped = self.id_rmap[dst_id]
                if self.get_node_obj(src_id_mapped) is None or self.get_node_obj(dst_id_mapped) is None:
                    continue
                ports_conf_all.append([src_id_mapped, dst_id_mapped, src_intf_id, dst_intf_id])

        return ports_conf_all


    def generate_and_install_paths(self, src_dst_pairs=None, max_paths=-1, use_rand_path=False):
        self.generate_paths(src_dst_pairs, max_paths, use_rand_path)
        # self.generate_paths_parallel(os.cpu_count(), src_dst_pairs)
        tracer.log("Paths have been generated")
        self.install_paths()
        tracer.log("Routes have been installed")

    def generate_and_install_paths_ensuring_ecmp_linearity(self, src_dst_pairs, m_paths):
        gen = path_generator_with_ecmp_hash_linearity()
        self.paths_all, trace_results_all = gen.generate_paths(self.G, src_dst_pairs, m_paths)
        tracer.log("Paths have been generated")
        self.install_paths()
        tracer.log("Routes have been installed")

    def get_num_endpoints(self):
        return self.n_ep

    @abstractmethod
    def create_topology(self):
        pass

    @abstractmethod
    def get_switch_ids(self):
        pass

    @abstractmethod
    def get_ep_ids(self):
        pass

    @abstractmethod
    def get_node_obj(self, node_id):
        pass

    def simulate_links_failed(self, links_failed):
        if len(links_failed) == 0:
            return

        self.G_original = copy.deepcopy(self.G)
        for node_id_from, node_id_to in links_failed:
            self.G.remove_edge(node_id_from, node_id_to)

    def simulate_net_recovery(self):
        if self.G_original is None:
            return
        self.G = copy.deepcopy(self.G_original)

    def bfs_paths(self, graph, start_node, dst_nodes=None, max_paths=-1, use_rand_path=False):
        paths = {}
        q = deque([(start_node, -1)])
        visited = set([start_node])
        reverse_path_tree = {}
        distances = {}
        distances[start_node] = 0

        while q:
            current_node, parent_node = q.popleft()
            if parent_node != -1:
                if current_node not in reverse_path_tree:
                    reverse_path_tree[current_node] = [parent_node]
                else:
                    reverse_path_tree[current_node].append(parent_node)
                distances[current_node] = distances[parent_node] + 1

            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, current_node))
                else:
                    if neighbor not in reverse_path_tree:
                        reverse_path_tree[neighbor] = [current_node]
                    else:
                        reverse_path_tree[neighbor].append(current_node)

        ep_id_set = None
        if dst_nodes is None:
            ep_id_set = set([ep.node_id for ep in self.endpoints if ep.node_id != start_node])
        else:
            ep_id_set = dst_nodes

        for ep_dst_id in ep_id_set:
            if ep_dst_id == start_node:
                continue
            paths[ep_dst_id] = []
            q = deque([(ep_dst_id, [ep_dst_id])])
            while q:
                cur_node_id, path = q.popleft()

                if cur_node_id in reverse_path_tree:
                    parent_node_ids = reverse_path_tree[cur_node_id]
                    if use_rand_path:
                        random.shuffle(parent_node_ids)

                    sufficient_paths = False
                    for parent_node_id in parent_node_ids:
                        if parent_node_id in path or parent_node_id in ep_id_set or \
                                distances[parent_node_id] >= distances[cur_node_id]:
                            continue
                        path_n = [parent_node_id] + path
                        if parent_node_id == start_node:
                            paths[ep_dst_id].append(path_n)
                            if 1 <= max_paths == len(paths[ep_dst_id]):
                                sufficient_paths = True
                                break
                        else:
                            q.append((parent_node_id, path_n))
                    if sufficient_paths:
                        break

        return paths

    def generate_paths(self, src_dst_flows=None, max_paths=-1, use_rand_path=False):
        self.paths_all = {}
        ep_id_list = None
        src_dst_map = {}
        if src_dst_flows != None:
            for flow in src_dst_flows:
                if flow.node_src_id not in src_dst_map:
                    src_dst_map[flow.node_src_id] = set()
                if flow.node_dst_id != flow.node_src_id:
                    src_dst_map[flow.node_src_id].add(flow.node_dst_id)
            ep_id_list = list(src_dst_map.keys())
        else:
            ep_id_list = [ep.node_id for ep in self.endpoints]
        n = len(ep_id_list)
        for i in range(n):
            ep_src_id = ep_id_list[i]
            paths = None
            if src_dst_flows != None:
                dst_nodes = src_dst_map[ep_src_id]
                paths = self.bfs_paths(self.G, ep_src_id, dst_nodes, max_paths, use_rand_path)
            else:
                paths = self.bfs_paths(self.G, ep_src_id, max_paths, use_rand_path)

            self.paths_all[ep_src_id] = paths
            tracer.log(f"==> Path generation progress: {i + 1} / {n}")


    def process_endpoint(self, ep_src_id, shared_paths_all, dst_nodes):
        paths = self.bfs_paths(self.G, ep_src_id, dst_nodes)
        shared_paths_all[ep_src_id] = paths
        return ep_src_id

    # def update_progress(self, result):
    #    global pbar
    #    pbar.update()

    def move_shared_path_all(self, n_endpoints, shared_paths_all, buf_ep_src_id, completion_event):
        i = 0
        while i < n_endpoints:
            ep_src_id = buf_ep_src_id.get()
            if ep_src_id == -1:
                break
            self.paths_all[ep_src_id] = shared_paths_all[ep_src_id]
            del shared_paths_all[ep_src_id]
            i += 1
            if self.display_move_th_progression:
                tracer.log(f"==> Progression for moving the shared paths: {i} / {n_endpoints}")
        completion_event.set()

    def generate_paths_parallel(self, n_processes, src_dst_pairs=None):
        ep_id_list = None
        src_dst_map = {}
        if src_dst_pairs != None:
            for pair in src_dst_pairs:
                if pair[0] not in src_dst_map:
                    src_dst_map[pair[0]] = set()
                src_dst_map[pair[0]].add(pair[1])
            ep_id_list = list(src_dst_pairs.keys())
        else:
            ep_id_list = [ep.node_id for ep in self.endpoints]

        n = len(ep_id_list)
        self.paths_all = {}
        self.display_move_th_progression = False

        with multiprocessing.Manager() as manager:
            buf_ep_src_id = multiprocessing.Queue()
            shared_paths_all = manager.dict()

            with tqdm(total=n, desc="Path generation progress") as pbar:
                with multiprocessing.Pool(processes=n_processes) as pool:
                    completion_event = threading.Event()
                    move_th = threading.Thread(target=self.move_shared_path_all,
                                               args=(n, shared_paths_all, buf_ep_src_id, completion_event))
                    move_th.start()

                    results = [pool.apply_async(self.process_endpoint,
                                                args=(ep_src_id,
                                                      shared_paths_all,
                                                      src_dst_map[ep_src_id] if src_dst_pairs is not None else None))
                               for ep_src_id in ep_id_list]

                    for result in results:
                        ep_src_id = result.get()
                        buf_ep_src_id.put(ep_src_id)
                        pbar.update()

                    self.display_move_th_progression = True

                    for _ in range(n):
                        buf_ep_src_id.put(-1)

                    completion_event.wait()
                    tracer.log(f"==> Progression for moving the shared paths: Finished")

        return

    def install_paths(self):
        total = int(np.sum([len(self.paths_all[k]) for k in self.paths_all]))
        k = 0
        for ep_src in self.paths_all.keys():
            for ep_dst in self.paths_all[ep_src].keys():
                if ep_src == ep_dst:
                    continue
                paths = self.paths_all[ep_src][ep_dst]
                for p in paths:
                    self.install_a_path(p)
                    # reverse routes
                    self.install_a_path(p[::-1])
                k += 1
                if np.mod(k, 100) == 0:
                    tracer.log(f"==> Route installing progression: {k} / {total}")

    def install_a_path(self, p):
        node_dst = p[len(p) - 1]
        node_dst_nei = p[len(p) - 2]
        node_dst_inf_addr = self.get_edge_addr(node_dst, node_dst_nei)
        for i in range(len(p) - 1):
            node_cur = p[i]
            node_nei = p[i + 1]
            node_obj = self.get_node_obj(node_cur)
            port_id = self.get_port_id(node_cur, node_nei)
            node_obj.routes.add_route(ip_dst_str=node_dst_inf_addr, port=port_id, metric=1)

    def read_routing_from_log_dir(self, log_prefix):
        if self.id_rmap is None or self.id_map is None:
            print("please call load_id_mapping_for_simai first")
            return None

        routing_csv_path = log_prefix + 'simai-routing'
        next_hop = defaultdict(lambda: defaultdict(list))

        with open(routing_csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                node_id, dst_id, next_id, next_node_type, interface_id = map(int, row)
                if next_node_type == 2:  # jump over the cases when the next hop is a NVswitch
                    continue
                node_id_mapped = self.id_rmap[node_id]
                dst_id_mapped = self.id_rmap[dst_id]
                next_id_mapped = self.id_rmap[next_id]
                next_hop[node_id_mapped][dst_id_mapped].append([next_id_mapped, interface_id])

        return next_hop

    def install_routes_from_csv(self, log_prefix):
        routing = self.read_routing_from_log_dir(log_prefix)

        for node_id in routing:
            node_obj = self.get_node_obj(node_id)
            if node_obj is None:
                continue # skip NVswitches
            for dst_id in routing[node_id]:
                dst = dst_id
                for [next_id, interface_id] in routing[node_id][dst_id]:
                    port_id = self.get_port_id(node_id, next_id)
                    if port_id != interface_id - 1:
                        print(f"Inconsistency between port IDs in HPS and SimAI with {port_id} and {interface_id} "
                              f"from node {node_id} to node {next_id}")
                    ip_dst_str = self.get_edge_addr(dst, list(self.G.neighbors(dst))[0])
                    node_obj.routes.add_route(ip_dst_str=ip_dst_str, port=port_id, metric=1)

        print(f"routes have been loaded from dir: {log_prefix}")

        ep_ids = self.get_ep_ids()
        self.compute_all_paths_from_routing_in_simai(routing, ep_ids)
        print("according to the routes, all_paths have been generated")

    def compute_all_paths_from_routing_in_simai(self, routing, ep_ids):
        self.paths_all = defaultdict(lambda: defaultdict(list))
        for src in ep_ids:
            for dst in ep_ids:
                if src == dst:
                    continue
                queue = deque()
                queue.append((src, [src]))
                while queue:
                    cur_node, path = queue.popleft()
                    if cur_node == dst:
                        self.paths_all[src][dst].append(path)
                        continue
                    if cur_node not in routing or dst not in routing[cur_node]:
                        continue
                    for next_id, _ in routing[cur_node][dst]:
                        if next_id not in path:
                            queue.append((next_id, path + [next_id]))

    def generate_ip_addresses(self):
        self.ip_addresses = []
        subnet_base = 10 << 24
        start_ip = subnet_base + 1

        current_ip = start_ip
        while len(self.ip_addresses) < self.n_links:
            octets = [(current_ip >> (8 * i)) & 0xFF for i in range(4)][::-1]
            ip_str = '.'.join(map(str, octets))

            if current_ip & 0xFF != 0 and current_ip & 0xFF != 255:
                self.ip_addresses.append(ip_str)

            current_ip += 1
            if (current_ip & 0xFF000000) != subnet_base:
                break

    def get_dst_addr(self, node_cur, node_dst):
        if node_cur not in self.paths_all or node_dst not in self.paths_all[node_cur]:
            return None
        p = self.paths_all[node_cur][node_dst]
        if len(p) == 0:
            return None
        p0 = p[0]
        addr = self.map_from_edge_to_port_addr[self.pair_hash(p0[len(p0) - 1], p0[len(p0) - 2])]
        return addr

    def get_neighbor_ids(self, node_id):
        neighbor_ids = [e[1] for e in list(self.G.edges(node_id))]
        return neighbor_ids

    def get_port_id(self, node_id, neighbor_id):
        if node_id < 0 or node_id >= self.n:
            return -1
        h = self.pair_hash(node_id, neighbor_id)
        if h not in self.map_from_edge_to_port_addr:
            return -1
        addr = self.map_from_edge_to_port_addr[h]
        node = self.get_node_obj(node_id)
        for port in node.ports.ports_info:
            if port.addr == addr:
                return port.id
        return -1

    def add_edge_addr(self, node_a_id, node_b_id, node_a_addr):
        self.map_from_edge_to_port_addr[self.pair_hash(node_a_id, node_b_id)] = node_a_addr

    def get_edge_addr(self, node_a_id, node_b_id):
        h = self.pair_hash(node_a_id, node_b_id)
        if h not in self.map_from_edge_to_port_addr:
            return None
        return self.map_from_edge_to_port_addr[h]

    def add_port(self, node_id, ip_addr):
        node_obj = self.get_node_obj(node_id)
        if node_obj == None:
            return -1
        port_id = node_obj.ports.add_port(ip_addr)
        return port_id

    def pair_hash(self, node_id_a, node_id_b):
        return node_id_a << 32 | node_id_b

    def clear_all_paths(self):
        self.paths_all = {}
        for node_id in range(self.n):
            node_obj = self.get_node_obj(node_id)
            node_obj.routes.routes_hash = {}

    def visualize(self):
        nx.draw(self.G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='#FFDEA2', linewidths=1,
                font_size=15)
        plt.show()

    def dump_ports(self):
        tracer.log("======== Endpoints ========")
        for node in self.endpoints:
            tracer.log("Node " + str(node.node_id) + ":\n" + str(self.get_node_obj(node.node_id).ports) + "\n")

        tracer.log("======== Leaf switches ========")
        for node in self.switches_leaf:
            tracer.log("Node " + str(node.node_id) + ":\n" + str(self.get_node_obj(node.node_id).ports) + "\n")

        tracer.log("======== Spine switches ========")
        for node in self.switches_spine:
            tracer.log("Node " + str(node.node_id) + ":\n" + str(self.get_node_obj(node.node_id).ports) + "\n")

    def dump_routes(self):
        tracer.log("======== Endpoints ========")
        for node in self.endpoints:
            tracer.log("Node " + str(node.node_id) + ":\n" + str(self.get_node_obj(node.node_id).routes) + "\n")

        tracer.log("======== Leaf switches ========")
        for node in self.switches_leaf:
            tracer.log("Node " + str(node.node_id) + ":\n" + str(self.get_node_obj(node.node_id).routes) + "\n")

        tracer.log("======== Spine switches ========")
        for node in self.switches_spine:
            tracer.log("Node " + str(node.node_id) + ":\n" + str(self.get_node_obj(node.node_id).routes) + "\n")



class sl_network(dcn_network):
    n_spine_sw = 0
    n_leaf_sw = 0
    n_ep_per_leaf_sw = 1

    switches_spine = None
    switches_leaf = None

    def __init__(self, n_spine_sw, n_leaf_sw, n_ep, n_ep_per_leaf_sw):
        self.G = nx.Graph()
        self.n_spine_sw = n_spine_sw
        self.n_leaf_sw = n_leaf_sw
        self.n_ep = n_ep
        self.n_ep_per_leaf_sw = n_ep_per_leaf_sw
        self.n = n_spine_sw + n_leaf_sw + n_ep
        self.switches_spine = []
        self.switches_leaf = []
        self.endpoints = []
        self.map_from_edge_to_port_addr = {}
        self.n_links = (self.n_spine_sw * self.n_leaf_sw + self.n_ep) * 2
        self.next_node_id = {}

    def create_topology(self):
        for i in range(self.n):
            self.G.add_node(i)
            if i < self.n_spine_sw:
                self.switches_spine.append(switch(i, self, layer_num=2))
            elif i < self.n_spine_sw + self.n_leaf_sw:
                self.switches_leaf.append(switch(i, self, layer_num=1))
            else:
                self.endpoints.append(endpoint(i, self, layer_num=0))

            self.next_node_id[i] = {}

        idx_ip = 0
        for spine_sw_id in range(self.n_spine_sw):
            for j in range(self.n_leaf_sw):
                leaf_sw_id = self.n_spine_sw + j
                self.G.add_edge(spine_sw_id, leaf_sw_id)
                port_spine_sw = self.add_port(spine_sw_id, self.get_ip_addresses(idx_ip, sw_id=spine_sw_id))
                port_leaf_sw = self.add_port(leaf_sw_id, self.get_ip_addresses(idx_ip + 1, sw_id=leaf_sw_id))
                self.next_node_id[spine_sw_id][port_spine_sw] = leaf_sw_id
                self.next_node_id[leaf_sw_id][port_leaf_sw] = spine_sw_id
                self.add_edge_addr(spine_sw_id, leaf_sw_id, self.get_ip_addresses(idx_ip, sw_id=spine_sw_id))
                self.add_edge_addr(leaf_sw_id, spine_sw_id, self.get_ip_addresses(idx_ip + 1, sw_id=leaf_sw_id))
                idx_ip += 2

        for j in range(self.n_leaf_sw):
            remain_ep = self.n_ep - j * self.n_ep_per_leaf_sw
            if remain_ep <= 0:
                break
            n_ep_leaf = np.min([self.n_ep_per_leaf_sw, remain_ep])
            for k in range(n_ep_leaf):
                leaf_sw_id = self.n_spine_sw + j
                ep_id = self.n_spine_sw + self.n_leaf_sw + j * self.n_ep_per_leaf_sw + k
                self.G.add_edge(leaf_sw_id, ep_id)
                port_leaf_sw = self.add_port(leaf_sw_id, self.get_ip_addresses(idx_ip, sw_id=leaf_sw_id))
                port_ep = self.add_port(ep_id, self.get_ip_addresses(idx_ip + 1, is_host=True, host_id=ep_id))
                self.next_node_id[leaf_sw_id][port_leaf_sw] = ep_id
                self.next_node_id[ep_id][port_ep] = leaf_sw_id
                self.add_edge_addr(leaf_sw_id, ep_id, self.get_ip_addresses(idx_ip, sw_id=leaf_sw_id))
                self.add_edge_addr(ep_id, leaf_sw_id, self.get_ip_addresses(idx_ip + 1, is_host=True, host_id=ep_id))
                idx_ip += 2

    def get_switch_ids(self):
        return [sw_j for sw_j in range(self.n_spine_sw + self.n_leaf_sw)]

    def get_ep_ids(self):
        return [j + self.n_spine_sw + self.n_leaf_sw for j in range(self.n_ep)]

    def get_node_obj(self, node_id):
        if node_id < self.n_spine_sw:
            return self.switches_spine[node_id]
        elif node_id < self.n_spine_sw + self.n_leaf_sw:
            return self.switches_leaf[node_id - self.n_spine_sw]
        elif node_id < self.n_spine_sw + self.n_leaf_sw + self.n_ep:
            return self.endpoints[node_id - self.n_spine_sw - self.n_leaf_sw]
        else:
            return None


class fattree_network(dcn_network):
    k = 0
    n_pods = 0
    n_cores = 0
    n_agg = 0
    n_edge = 0
    n_hosts = 0
    switches_core = None
    switches_agg = None
    switches_edge = None

    def __init__(self, k):
        self.k = k  # Number of ports per switch
        self.n_pods = k
        self.n_cores = k ** 2 // 4
        self.n_agg = k ** 2 // 2
        self.n_edge = k ** 2 // 2
        self.n_hosts = k ** 3 // 4
        self.n = self.n_cores + self.n_agg + self.n_edge + self.n_hosts
        self.G = nx.Graph()
        self.switches_core = []
        self.switches_agg = []
        self.switches_edge = []
        self.endpoints = []
        self.map_from_edge_to_port_addr = {}
        self.next_node_id = {}

        self.n_links = k ** 3 * 3 // 2
        self.n_ep = self.n_hosts

    def create_topology(self):
        # Initialize nodes and their entries in next_node_id
        for i in range(self.n):
            self.G.add_node(i)
            self.next_node_id[i] = {}  # Initialize each node in next_node_id
            # Initialize node sets
            if i < self.n_cores:
                self.switches_core.append(switch(i, self, layer_num=3))
            elif i < self.n_cores + self.n_pods * (self.k // 2):
                self.switches_agg.append(switch(i, self, layer_num=2))
            elif i < self.n_cores + self.n_pods * (self.k // 2) + self.n_pods * (self.k // 2):
                self.switches_edge.append(switch(i, self, layer_num=1))
            else:
                self.endpoints.append(endpoint(i, self, layer_num=0))

        # Connect core to aggregation switches
        idx_ip = 0
        for core_id in range(self.n_cores):
            for pod in range(self.n_pods):
                agg_id = self.n_cores + pod * (self.k // 2) + core_id // (self.k // 2)
                self.G.add_edge(core_id, agg_id)
                port_core = self.add_port(core_id, self.get_ip_addresses(idx_ip, sw_id=core_id))
                port_agg = self.add_port(agg_id, self.get_ip_addresses(idx_ip + 1, sw_id=agg_id))
                self.next_node_id[core_id][port_core] = agg_id
                self.next_node_id[agg_id][port_agg] = core_id
                self.add_edge_addr(core_id, agg_id, self.get_ip_addresses(idx_ip, sw_id=core_id))
                self.add_edge_addr(agg_id, core_id, self.get_ip_addresses(idx_ip + 1, sw_id=agg_id))
                idx_ip += 2

        # Connect aggregation to edge switches
        for pod in range(self.n_pods):
            agg_offset = self.n_cores + pod * (self.k // 2)
            edge_offset = agg_offset + self.n_pods * (self.k // 2)
            for i in range(self.k // 2):
                for j in range(self.k // 2):
                    agg_id = agg_offset + i
                    edge_id = edge_offset + j
                    self.G.add_edge(agg_id, edge_id)
                    port_agg = self.add_port(agg_id, self.get_ip_addresses(idx_ip, sw_id=agg_id))
                    port_edge = self.add_port(edge_id, self.get_ip_addresses(idx_ip + 1, sw_id=edge_id))
                    self.next_node_id[agg_id][port_agg] = edge_id
                    self.next_node_id[edge_id][port_edge] = agg_id
                    self.add_edge_addr(agg_id, edge_id, self.get_ip_addresses(idx_ip, sw_id=agg_id))
                    self.add_edge_addr(edge_id, agg_id, self.get_ip_addresses(idx_ip + 1, sw_id=edge_id))
                    idx_ip += 2

        # Connect edge switches to hosts
        for pod in range(self.n_pods):
            edge_offset = self.n_cores + self.n_agg + pod * (self.k // 2)
            for i in range(self.k // 2):
                edge_id = edge_offset + i
                host_offset = self.n_cores + self.n_agg + self.n_edge + pod * (self.k // 2) ** 2 + i * (self.k // 2)
                for j in range(self.k // 2):
                    host_id = host_offset + j
                    self.G.add_edge(edge_id, host_id)
                    port_edge = self.add_port(edge_id, self.get_ip_addresses(idx_ip, sw_id=edge_id))
                    port_host = self.add_port(host_id, self.get_ip_addresses(idx_ip + 1, sw_id=host_id))
                    self.next_node_id[edge_id][port_edge] = host_id
                    self.next_node_id[host_id][port_host] = edge_id
                    self.add_edge_addr(edge_id, host_id, self.get_ip_addresses(idx_ip, sw_id=edge_id))
                    self.add_edge_addr(host_id, edge_id, self.get_ip_addresses(idx_ip + 1, is_host=True, host_id=host_id))
                    idx_ip += 2

        tracer.log(self.n)

    def get_switch_ids(self):
        return [sw.node_id for sw in self.switches_core + self.switches_agg + self.switches_edge]

    def get_ep_ids(self):
        return [ep.node_id for ep in self.endpoints]

    def get_node_obj(self, node_id):
        if node_id < len(self.switches_core):
            return self.switches_core[node_id]
        elif node_id < len(self.switches_core) + len(self.switches_agg):
            return self.switches_agg[node_id - len(self.switches_core)]
        elif node_id < len(self.switches_core) + len(self.switches_agg) + len(self.switches_edge):
            return self.switches_edge[node_id - len(self.switches_core) - len(self.switches_agg)]
        elif node_id < len(self.switches_core) + len(self.switches_agg) + len(self.switches_edge) + len(
                self.endpoints):
            return self.endpoints[
                node_id - len(self.switches_core) - len(self.switches_agg) - len(self.switches_edge)]
        else:
            return None


class aspen_trees_network(dcn_network):
    n = 0
    n_ep = 0
    n_links = 0
    next_node_id = None
    endpoints = None

    n_levels = 0
    k_port_per_switch = 0
    S_switches_per_level = 0
    p = None
    m = None
    r = None
    c = None
    c_factored = None
    pods_at_diff_levels = None
    switches_of_pods = None
    switch_ids = None
    ep_ids = None
    node_obj_map = None

    # c_factored is a sequence of factors of k, each representing the choice of c at the n-th layer to the 2nd layer
    def __init__(self, n_levels, k_port_per_switch, c_factored=None):
        self.n_levels = n_levels
        self.k_port_per_switch = k_port_per_switch
        self.next_node_id = {}
        self.pods_at_diff_levels = {}
        self.switches_of_pods = {}
        self.endpoints = []
        if c_factored == None or len(c_factored) == 0:
            factored = factorint(k_port_per_switch)
            self.c_factored = []
            for k in factored:
                self.c_factored += [k] * factored[k]
            self.c_factored.sort()
        else:
            self.c_factored = c_factored

        self.G = nx.Graph()
        self.switch_ids = []
        self.ep_ids = []
        self.node_obj_map = {}

        self.map_from_edge_to_port_addr = {}
        self.next_node_id = {}

        res = self.create_aspen_topo_parameters()
        if res == -1:
            return

        tracer.log(f"p: {self.p}, m: {self.m}, r: {self.r}, c: {self.c}, n_links: {self.n_links}")

    def create_topology(self):
        i = self.n_levels - 1
        node_id = 0
        pod_id = 0

        while i >= 0:
            self.pods_at_diff_levels[i] = [pod_id + pod_seq for pod_seq in range(self.p[i])]
            pod_id += self.p[i]
            for pod_id_it in self.pods_at_diff_levels[i]:
                switches_at_pod = []
                for j in range(self.m[i]):
                    self.G.add_node(node_id)
                    self.next_node_id[node_id] = {}
                    sw_obj = switch(node_id, self, layer_num=i + 1)
                    switches_at_pod.append(sw_obj)
                    self.switch_ids.append(node_id)
                    self.node_obj_map[node_id] = sw_obj
                    node_id += 1
                self.switches_of_pods[pod_id_it] = switches_at_pod

            i -= 1

        for j in range(self.n_ep):
            self.G.add_node(node_id)
            self.next_node_id[node_id] = {}
            ep_obj = endpoint(node_id, self, layer_num=0)
            self.endpoints.append(ep_obj)
            self.ep_ids.append(node_id)
            self.node_obj_map[node_id] = ep_obj
            node_id += 1

        idx_ip = 0
        i = self.n_levels - 1
        while i >= 1:
            u = 0
            for pod_id_level_i in self.pods_at_diff_levels[i]:
                for v in range(u * self.r[i], (u + 1) * self.r[i]):
                    pod_id_level_i_minus_1 = self.pods_at_diff_levels[i - 1][v]
                    w = 0
                    for t in range(self.c[i]):
                        for switch_level_i in self.switches_of_pods[pod_id_level_i]:
                            switch_id_level_i = switch_level_i.node_id
                            switch_level_i_minus_1 = \
                                self.switches_of_pods[pod_id_level_i_minus_1][w // (self.k_port_per_switch // 2)]

                            switch_id_level_i_minus_1 = switch_level_i_minus_1.node_id
                            self.G.add_edge(switch_id_level_i, switch_id_level_i_minus_1)
                            port_level_i = self.add_port(switch_id_level_i, self.get_ip_addresses(idx_ip, sw_id=switch_id_level_i))
                            port_level_minus_i = self.add_port(switch_id_level_i_minus_1,
                                                               self.get_ip_addresses(idx_ip + 1, sw_id=switch_id_level_i_minus_1))

                            self.next_node_id[switch_id_level_i][port_level_i] = switch_id_level_i_minus_1
                            self.next_node_id[switch_id_level_i_minus_1][port_level_minus_i] = switch_id_level_i
                            self.add_edge_addr(switch_id_level_i, switch_id_level_i_minus_1,
                                               self.get_ip_addresses(idx_ip, sw_id=switch_id_level_i))
                            self.add_edge_addr(switch_id_level_i_minus_1, switch_id_level_i,
                                               self.get_ip_addresses(idx_ip + 1, sw_id=switch_id_level_i_minus_1))

                            # tracer.log(f"edge: {switch_id_level_i}, {switch_id_level_i_minus_1}")

                            idx_ip += 2
                            w += 1
                u += 1
            i -= 1

        ep_idx_base = 0
        for pod_id_edge in self.pods_at_diff_levels[0]:
            for switch_edge in self.switches_of_pods[pod_id_edge]:
                switch_id_edge = switch_edge.node_id
                for sw_port_j in range(self.k_port_per_switch // 2):
                    ep_idx = ep_idx_base + sw_port_j
                    ep_id = self.endpoints[ep_idx].node_id
                    self.G.add_edge(switch_id_edge, ep_id)
                    port_edge = self.add_port(switch_id_edge, self.get_ip_addresses(idx_ip, sw_id=switch_id_edge))
                    port_ep = self.add_port(ep_id, self.get_ip_addresses(idx_ip + 1, is_host=True, host_id=ep_id))
                    self.next_node_id[switch_id_edge][port_edge] = ep_id
                    self.next_node_id[ep_id][port_ep] = switch_id_edge
                    self.add_edge_addr(switch_id_edge, ep_id, self.get_ip_addresses(idx_ip, sw_id=switch_id_edge))
                    self.add_edge_addr(ep_id, switch_id_edge, self.get_ip_addresses(idx_ip + 1, is_host=True, host_id=ep_id))

                    # tracer.log(f"edge: {switch_id_edge}, {ep_id}")
                    idx_ip += 2

                ep_idx_base += self.k_port_per_switch // 2

    def create_aspen_topo_parameters(self):
        self.p = [0] * self.n_levels
        self.m = [0] * self.n_levels
        self.r = [0] * self.n_levels
        self.c = [0] * self.n_levels
        self.n_ep = 0
        self.n_links = 0

        i = self.n_levels - 1

        n_factors = len(self.c_factored)
        idx_factor = 0
        self.p[self.n_levels - 1] = 1
        downlinks = self.k_port_per_switch
        while i >= 1:
            if idx_factor < n_factors:
                self.c[i] = self.c_factored[idx_factor]
                idx_factor += 1
            else:
                self.c[i] = 1
            self.r[i] = int(downlinks / self.c[i])
            self.p[i - 1] = self.p[i] * self.r[i]
            downlinks = int(self.k_port_per_switch / 2)
            i = i - 1

        self.S_switches_per_level = self.p[0] * self.c[1]
        self.m[self.n_levels - 1] = int(self.S_switches_per_level / 2)
        for i in range(0, self.n_levels - 1):
            self.m[i] = int(self.S_switches_per_level / self.p[i])
            if self.m[i] != int(self.m[i]):
                tracer.log("an error occurs in creating aspen trees topology")
                return -1
        if self.m[self.n_levels - 1] != int(self.m[self.n_levels - 1]):
            tracer.log("an error occurs in creating aspen trees topology")
            return -1

        self.n_ep = self.S_switches_per_level * self.k_port_per_switch // 2
        self.n = self.S_switches_per_level // 2 + self.S_switches_per_level * (self.n_levels - 1) + self.n_ep
        self.n_links = (self.p[self.n_levels - 1] * self.m[self.n_levels - 1] * \
                        self.r[self.n_levels - 1] * self.c[self.n_levels - 1] * self.n_levels) * 2

        return 0

    def get_switch_ids(self):
        return self.switch_ids

    def get_ep_ids(self):
        return self.ep_ids

    def get_node_obj(self, node_id):
        if node_id not in self.node_obj_map:
            return None
        return self.node_obj_map[node_id]

class packet:
    ip_src = "192.168.0.1"
    ip_dst = "192.168.0.2"
    port_src = 6000
    port_dst = 6001
    ttl = 255

    def to_seq(self):
        r = bytearray(socket.inet_aton(self.ip_src))
        r.extend(socket.inet_aton(self.ip_dst))
        r.extend(self.port_src.to_bytes(2, byteorder='big'))
        r.extend(self.port_dst.to_bytes(2, byteorder='big'))
        return r

    def to_seq_with_delta(self, delta):
        port_src_xored = self.port_src ^ delta
        r = bytearray(socket.inet_aton(self.ip_src))
        r.extend(socket.inet_aton(self.ip_dst))
        r.extend(port_src_xored.to_bytes(2, byteorder='big'))
        r.extend(self.port_dst.to_bytes(2, byteorder='big'))
        return r

    @classmethod
    def get_packet_zero(self):
        r = bytearray([0] * pkt_header_size)
        return r

    @classmethod
    def get_delta(self, delta):
        b = bytearray([0] * pkt_header_size)
        de = delta.to_bytes(2, byteorder='big')
        b[8] = de[0]
        b[9] = de[1]
        return b

    def __repr__(self):
        return f"a packet with ip_src: {self.ip_src}, ip_dst: {self.ip_dst}, " \
               f"proto: {self.proto}, port_src: {self.port_src}, port_dst: {self.port_dst}"


class endpoint:
    node_id = -1
    ports = None
    routes = None
    net = None

    def __init__(self, node_id, net, layer_num=-1):
        self.node_id = node_id
        self.ports = port_table()
        self.routes = route_table()
        self.net = net
        self.layer_num = layer_num

    def send_test_packet(self, port_src, node_id_dst, port_dst, ttl=255):
        pkt = packet()

        addr = self.net.get_dst_addr(self.node_id, node_id_dst)
        if addr == None:
            tracer.log(f"[Node {self.node_id}] error to get the address of dst node {node_id_dst}")
        pkt.ip_src = self.ports.ports_info[0].addr
        pkt.ip_dst = addr
        pkt.port_src = port_src
        pkt.port_dst = port_dst
        pkt.ttl = ttl

        path_fd = self.send_packet(pkt)
        return path_fd

    def send_packet(self, pkt: packet):
        m = self.routes.match_route(pkt.ip_dst)
        if len(m) == 0:
            tracer.log(f"[Node {self.node_id}] no route to dst node")
            return

        next_node_id = self.net.next_node_id[self.node_id][m[0][0]]
        next_node = self.net.get_node_obj(next_node_id)

        tracer.log(f"[Node {self.node_id}] forward pkt to next hop {next_node_id}")
        path_fd = next_node.receive_packet(pkt)
        return path_fd

    def receive_packet(self, pkt: packet):
        tracer.log(f"[Node {self.node_id}] received packet, src={pkt.ip_src}, dst={pkt.ip_dst}")
        return [self.node_id]


class EcmpHashAlgorithm(Enum):
    CRC = 0
    XOR = 1
    CRC_32LO = 1
    CRC_32HI = 2
    CRC_CCITT = 3
    CRC_XOR = 4


class HashingFunc:
    @staticmethod
    def ecmp_hash_crc(data: bytearray) -> int:
        return zl.crc32(data)

    @staticmethod
    def ecmp_hash_xor(data: bytearray) -> int:
        result = 0
        for b in data:
            result ^= b
        return result

    @staticmethod
    def ecmp_hash_crc32_lo(data: bytearray) -> int:
        return zl.crc32(data) & 0xFFFF

    @staticmethod
    def ecmp_hash_crc32_hi(data: bytearray) -> int:
        return (zl.crc32(data) >> 16) & 0xFFFF

    @staticmethod
    def ecmp_hash_crc_ccitt(data: bytearray) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc

    @staticmethod
    def ecmp_hash_crc_xor(data: bytearray) -> int:
        return zl.crc32(data) ^ HashingFunc.ecmp_hash_xor(data)

    @classmethod
    def compute_hash(cls, data: bytearray, algo: EcmpHashAlgorithm) -> int:
        hash_function_map = {
            EcmpHashAlgorithm.CRC: cls.ecmp_hash_crc,
            EcmpHashAlgorithm.XOR: cls.ecmp_hash_xor,
            EcmpHashAlgorithm.CRC_32LO: cls.ecmp_hash_crc32_lo,
            EcmpHashAlgorithm.CRC_32HI: cls.ecmp_hash_crc32_hi,
            EcmpHashAlgorithm.CRC_CCITT: cls.ecmp_hash_crc_ccitt,
            EcmpHashAlgorithm.CRC_XOR: cls.ecmp_hash_crc_xor,
        }
        return hash_function_map[algo](data)

class switch:
    target_simai = False
    node_id = -1
    ports = None
    routes = None
    n_ports = 16
    seed = None
    permutation = None
    hv_buckets_nnhs = None
    net = None

    def __init__(self, node_id, net, seed=None, permutation=None, layer_num=-1):
        if seed is None:
            self.seed = bytearray(random.randint(0, 255) for _ in range(pkt_header_size))
        else:
            self.seed = seed

        if permutation is None:
            self.permutation = random.sample(range(pkt_header_size), pkt_header_size)
        else:
            self.permutation = permutation

        self.hv_buckets_nnhs = {}
        self.node_id = node_id
        self.ports = port_table()
        self.routes = route_table()
        self.net = net
        self.hashing_alg = EcmpHashAlgorithm.CRC
        self.layer_num = layer_num

    def receive_packet(self, pkt: packet):
        next_node_id = self.ecmp_decide(pkt)
        if next_node_id == -1:
            tracer.log(f"[Node {self.node_id}] drop pkt due to no route")
            return [-1]

        pkt.ttl -= 1
        if pkt.ttl > 0:
            next_node = self.net.get_node_obj(next_node_id)
            tracer.log(f"[Node {self.node_id}] forward pkt to next hop {next_node_id} with ttl {pkt.ttl}")
            fb = next_node.receive_packet(pkt)
            return [self.node_id] + fb if fb != -1 else [self.node_id]
        else:
            tracer.log(f"[Node {self.node_id}] drops pkt due to ttl 0")
            return [self.node_id]

    def ecmp_decide(self, pkt: packet):
        hv, port_egress = self.ecmp_hash_by_packet(pkt)
        next_node_id = self.net.next_node_id[self.node_id][port_egress]
        return next_node_id

    def ecmp_hash_by_packet(self, pkt: packet):
        m = self.routes.match_route(pkt.ip_dst)

        if switch.target_simai:
            m_ = sorted(m, key=lambda x: x[0])
        else:
            m_ = m

        n_nexthop = len(m_)
        if n_nexthop == 0:
            return -1

        ba = pkt.to_seq()
        hv = self.ecmp_hash_by_ba(ba, n_nexthop)
        port_egress = m_[hv][0]
        return (hv, port_egress)

    def ecmp_hash_by_ba(self, ba: bytearray, n_nexthop):
        ba_perm = bytearray(len(ba))
        for i, perm_idx in enumerate(self.permutation):
            ba_perm[i] = ba[perm_idx]

        s = bytearray()
        for b1, b2 in zip(ba_perm, self.seed):
            s.append(b1 ^ b2)
        #h = zl.crc32(s)
        h = HashingFunc.compute_hash(s, self.hashing_alg)
        hv = h % n_nexthop

        # if n_nexthop not in self.hv_buckets_nnhs:
        #    self.hv_buckets_nnhs[n_nexthop] = random.sample(range(n_nexthop), n_nexthop)

        # hv_b = self.hv_buckets_nnhs[n_nexthop][hv]

        return hv
