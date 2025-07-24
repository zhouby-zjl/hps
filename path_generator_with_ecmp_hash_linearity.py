from collections import deque
import numpy as np
import random
import math
from tracer import tracer
from collections import defaultdict
import time

class path_generator_with_ecmp_hash_linearity:
    class ptree:
        def __init__(self, start_node):
            self.parents = {}
            self.start_node = start_node

    class ctree_info:
        def __init__(self):
            self.conn = {}
            self.conn_nl = {}
            #self.links_num = {}
            self.downs = {}
            self.conn_all = {} # for the tracing purpose
            self.conn_nl_all = {} # for the tracing purpose

    def generate_paths(self, G, src_dst_flows,
                       m_paths, time_records = None, trace=False):
        paths_all = {}
        src_dst_map = {}

        for flow in src_dst_flows:
            if flow.node_src_id not in src_dst_map:
                src_dst_map[flow.node_src_id] = set()
            src_dst_map[flow.node_src_id].add(flow.node_dst_id)
        ep_id_list = list(src_dst_map.keys())
        n = len(ep_id_list)

        #trace_results_all = None
        if trace:
            #trace_results_all = []
            link_density_map_paths_pow2 = {}
            link_density_map_paths_all = {}
            rec_all = []
        else:
            link_density_map_paths_pow2 = None
            link_density_map_paths_all = None
            rec_all = None

        for i in range(n):
            ep_src_id = ep_id_list[i]
            if src_dst_flows != None:
                dst_nodes = src_dst_map[ep_src_id]
            else:
                dst_nodes = None

            start_time_ns = time.time() * 1e9
            paths = self.bfs_paths_with_ecmp_hash_linearity(G, ep_src_id, m_paths, dst_nodes, trace=trace,
                                                            link_density_map_paths_pow2=link_density_map_paths_pow2,
                                                            link_density_map_paths_all=link_density_map_paths_all,
                                                            rec_all=rec_all)
            execution_time_ns = int(time.time() * 1e9 - start_time_ns)
            if time_records is not None:
                time_records.append(execution_time_ns)
            paths_all[ep_src_id] = paths
            #tracer.log(f"==> Path generation progress: {i + 1} / {n}")

            #if trace:
            #    trace_results_all.append(trace_results)

        if trace:
            trace_results_all = {'link_density_map_paths_pow2': link_density_map_paths_pow2,
                                 'link_density_map_paths_all': link_density_map_paths_all,
                                 'rec_all': rec_all}
            return paths_all, trace_results_all
        else:
            return paths_all, None

    def is_power_of_two(self, n):
        """Check if a number is a power of two."""
        return n > 0 and (n & (n - 1)) == 0

    def check_ecmp_hash_linearity(self, paths):
        hop_next_hop_map = defaultdict(set)

        for path in paths:
            for i in range(len(path) - 1):
                hop = path[i]
                next_hop = path[i + 1]
                hop_next_hop_map[hop].add(next_hop)

        results = {}
        for hop, next_hops in hop_next_hop_map.items():
            count = len(next_hops)
            if self.is_power_of_two(count):
                continue

            results[hop] = {
                'next_hops': next_hops,
                'count': count,
                'power_of_two': self.is_power_of_two(count)
            }

        # Report hops that violate ECMP hash linearity
        #print("Hops that do NOT have a power-of-two number of next hops:")
        #for hop, info in results.items():
        #    if not info['power_of_two']:
        #        print(f"Hop {hop}: {info['count']} next hops → {info['next_hops']}")

        pu = len(results.keys()) / len(hop_next_hop_map.items())

        return results, pu

    def bfs_paths_with_ecmp_hash_linearity(self, graph, start_node, m_paths, dst_nodes=None, trace=False,
                                           link_density_map_paths_pow2={}, link_density_map_paths_all={},
                                           rec_all=[]):
        pt = self.construct_reverse_path_tree(start_node, graph)

        if dst_nodes is None:
            ep_id_set = set([ep.node_id for ep in self.endpoints if ep.node_id != start_node])
        else:
            ep_id_set = dst_nodes

        paths = {}
        #if trace:
        #    link_density_map_paths_pow2 = {}
        #    link_density_map_paths_all = {}
        #    rec_all = []

        for dst_node in ep_id_set:
            if dst_node == start_node:
                continue
            ctree = self.construct_ctree(start_node, dst_node, pt, trace=trace)
            if m_paths >= 1:
                m_paths_to_generate = int(np.min([ctree.conn[start_node], m_paths]))
            else:
                m_paths_to_generate = ctree.conn[start_node]
            T_r, N_left_total = self.initially_allocate_paths(ctree, start_node, m_paths_to_generate)
            T_r = self.reallocate_remaining_paths(ctree, start_node, T_r, N_left_total)
            paths_to_dst = self.gen_paths(start_node, T_r)
            paths[dst_node] = paths_to_dst

            if trace:
                rec = [ctree.conn[start_node], ctree.conn_all[start_node],
                       m_paths_to_generate, m_paths_to_generate - N_left_total, N_left_total]
                rec_all.append(rec)

                paths_all_to_dst = []
                q = deque([(start_node, [start_node])])
                while q:
                    cur_hop, path = q.popleft()
                    if cur_hop not in ctree.downs:
                        paths_all_to_dst.append(path)
                        continue
                    for next_hop in ctree.downs[cur_hop]:
                        q.append((next_hop, path + [next_hop]))

                self.count_path_density_for_links(paths_all_to_dst, link_density_map_paths_all)
                self.count_path_density_for_links(paths_to_dst, link_density_map_paths_pow2)

        #trace_result = None
        #if trace:
        #    trace_result = {'link_density_map_paths_pow2': link_density_map_paths_pow2,
        #                    'link_density_map_paths_all': link_density_map_paths_all,
        #                    'rec_all': rec_all}
        return paths#, trace_result

    def count_path_density_for_links(self, paths, link_density_map):
        for p in paths:
            for i in range(len(p) - 1):
                l_id = (p[i], p[i + 1])
                if l_id not in link_density_map:
                    link_density_map[l_id] = 1
                else:
                    link_density_map[l_id] += 1


    def construct_ctree(self, start_node, dst_node, pt, trace=False):
        ctree = self.ctree_info()
        ctree.conn[dst_node] = 1
        ctree.conn_nl[dst_node] = [1]
        if trace:
            ctree.conn_all[dst_node] = 1
            ctree.conn_nl_all[dst_node] = [1]

        q = deque([(dst_node, -1)])
        visited = set([dst_node])

        while q:
            cur_node, parent_node = q.popleft()

            if cur_node not in ctree.conn:
                links = self.most_significant_bit(len(ctree.conn_nl[cur_node]))
                ctree.conn[cur_node] = sum(sorted(ctree.conn_nl[cur_node], reverse=True)[:links])
                if trace:
                    ctree.conn_all[cur_node] = sum(ctree.conn_nl_all[cur_node])

            if cur_node == start_node:
                break

            for neighbor in pt.parents[cur_node]:
                ctree.conn_nl.setdefault(neighbor, []).append(ctree.conn[cur_node])
                if trace:
                    ctree.conn_nl_all.setdefault(neighbor, []).append(ctree.conn_all[cur_node])
                ctree.downs.setdefault(neighbor, []).append(cur_node)
                if neighbor in visited:
                    continue
                q.append((neighbor, cur_node))
                visited.add(neighbor)

        return ctree

    def construct_reverse_path_tree(self, start_node, graph):
        q = deque([(start_node, -1)])
        visited = {start_node}
        pt = self.ptree(start_node)

        while q:
            current_node, parent_node = q.popleft()

            if parent_node != -1:
                pt.parents.setdefault(current_node, []).append(parent_node)

            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, current_node))
                elif neighbor != parent_node and neighbor not in pt.parents[current_node]:
                    pt.parents.setdefault(neighbor, []).append(current_node)

        return pt

    def initially_allocate_paths(self, cinfo, start_node, N):
        q = deque([(start_node, -1, N)])
        T_r = {}
        N_left_total = 0

        # Allocate paths, and track and remove over-allocations
        while q:
            cur_node_id, parent_node_id, N_comp = q.popleft()
            if cur_node_id not in cinfo.downs:
                continue
            C_comp = cinfo.conn_nl[cur_node_id]

            # generate a subset (C_s) of C_comp, to make the sum of the subset no less than N_comp
            # N_comp - N_left must be powers of 2
            # return the C_s and the indexes of C_s_idx within C_comp, where N_left is equal to N_comp - sum(C_s)
            C_s, C_s_idx, N_left = self.find_random_subset_gte_power_of_2(C_comp, N_comp)
            if C_s is None:
                continue
            # Generate the random choices (N_s) within C_s, to make the sum of N_s being N_comp - N_left
            N_s = self.generate_random_limited_sum(N_comp - N_left, C_s)
            # Generate the down node IDs selected according to C_s_idx
            #print(f"cur_node_id: {cur_node_id}, C_s: {C_s}, N_s: {N_s}\n")

            down_ids_sel = [cinfo.downs[cur_node_id][idx] for idx in C_s_idx]
            T_r[cur_node_id] = [down_ids_sel, N_s, parent_node_id]
            q.extend((down_id_sel, cur_node_id, n) for down_id_sel, n in zip(down_ids_sel, N_s))

            if N_left == 0:
                continue

            while parent_node_id != -1:
                node_ids_nl_cur = T_r[parent_node_id][0]
                for i, down_id in enumerate(node_ids_nl_cur):
                    if down_id == cur_node_id:
                        T_r[parent_node_id][1][i] -= N_left
                        break
                cur_node_id = parent_node_id
                parent_node_id = T_r[parent_node_id][2]

            N_left_total += N_left


        return T_r, N_left_total

    def reallocate_remaining_paths(self, cinfo, start_node, T_r, N_left_total):
        q = deque([(start_node, -1)])
        while q and N_left_total > 0:
            cur_node_id, parent_node_id = q.popleft()
            node_ids_nl_all = cinfo.downs.get(cur_node_id, [])
            node_ids_nl_sel = T_r[cur_node_id][0]
            n_sel = len(node_ids_nl_sel)
            m = np.min([int(np.log2(N_left_total + n_sel)), int(math.log2(len(node_ids_nl_all)))])
            n_opt = 2**m - n_sel

            for i in range(n_opt):
                diverging_node_id = cur_node_id
                diverging_par_node_id = parent_node_id
                diverging_down_node_ids_to_sel = [id for id in cinfo.downs[diverging_node_id] if id not in T_r[diverging_node_id][0]]
                while diverging_node_id in cinfo.downs:
                    diverging_down_node_id = random.choice(diverging_down_node_ids_to_sel)
                    if diverging_node_id in T_r and diverging_down_node_id not in T_r[diverging_node_id][0]: # overlapped with an existing path
                        T_r[diverging_node_id][0].append(diverging_down_node_id)
                        T_r[diverging_node_id][1].append(1)
                    elif diverging_node_id not in T_r:
                        T_r[diverging_node_id] = [[diverging_down_node_id], [1], diverging_par_node_id]
                    else:
                        break
                    diverging_par_node_id = diverging_node_id
                    diverging_node_id = diverging_down_node_id
                    diverging_down_node_ids_to_sel = cinfo.downs[diverging_node_id]

            N_left_total -= n_opt

            node_ids_nl_sel = T_r[cur_node_id][0]
            for node_id_nl in node_ids_nl_sel:
                if node_id_nl not in T_r:
                    continue
                q.append((node_id_nl, cur_node_id))
            #q.extend((node_id_nl, cur_node_id) for node_id_nl in node_ids_nl_sel)

        return T_r

    def find_random_subset_gte_power_of_2(self, C, N, max_attempts=10000):
        if not self.can_find_valid_subset(C, N):
            print(f"No valid subset can be found where the sum >= {N}")
            return None, None, None

        for attempt in range(max_attempts):
            C_sfl, C_idx = self.shuffle_with_indices(C)
            max_size = len(C_sfl)
            if max_size > N:
                N_r = 2**int(math.log2(N))
                target_size = N_r
            else:
                possible_sizes = [2 ** i for i in range(int(math.log2(max_size)) + 1)]
                target_size = random.choice(possible_sizes)
                N_r = N
            C_chosen = []
            C_chosen_idx = []
            C_sum = 0

            for i in range(target_size):
                c = C_sfl[i]
                C_chosen.append(c)
                C_chosen_idx.append(C_idx[i])
                C_sum += c

            if C_sum >= N_r:
                return C_chosen, C_chosen_idx, N - N_r

        print(f"Failed to find a valid subset after {max_attempts} attempts")
        return None, None, None

    def generate_random_limited_sum(self, N, C):
        k = len(C)

        if sum(C) < N or k > N:
            return None

        N_sel = [1] * k
        remaining_sum = N - k
        remaining_allowance = [c_i - 1 for c_i in C]

        available_indices = [i for i in range(k) if remaining_allowance[i] > 0]
        while remaining_sum > 0 and available_indices:
            i = random.choice(available_indices)
            N_sel[i] += 1
            remaining_allowance[i] -= 1
            remaining_sum -= 1
            if remaining_allowance[i] == 0:
                available_indices.remove(i)

        return N_sel

    def create_diverging_paths(self, cur_node_id, n_opt, parent_node_id, r, cinfo):
        for i in range(n_opt):
            node_id = cur_node_id
            par_node_id = parent_node_id
            init_down_node_ids = [id for id in cinfo.downs[node_id] if id not in r[node_id][0]]
            while node_id in cinfo.downs:
                if node_id == cur_node_id:
                    down_node_ids_to_sel = init_down_node_ids
                else:
                    down_node_ids_to_sel = cinfo.downs[node_id]
                down_node_id = random.choice(down_node_ids_to_sel)
                if node_id in r:
                    if down_node_id not in r[node_id][0]:
                        r[node_id][0].append(down_node_id)
                        r[node_id][1].append(1)
                    else:
                        break
                else:
                    r[node_id] = [[down_node_id], [1], par_node_id]
                par_node_id = node_id
                node_id = down_node_id

    def gen_paths(self, start_node, r):
        paths_to_dst = []
        q = deque([(start_node, [start_node])])
        while q:
            cur_node, path = q.popleft()
            for down_node in r[cur_node][0]:
                path_n = path + [down_node]
                if down_node not in r:
                    paths_to_dst.append(path_n)
                else:
                    q.append((down_node, path_n))
        return paths_to_dst

    def most_significant_bit(self, n):
        if n == 0:
            return 0

        msb = 0
        while n > 1:
            n = n >> 1
            msb += 1
        return 1 << msb

    def can_find_valid_subset(self, C, N):
        sorted_C = sorted(C, reverse=True)

        for i in range(int(math.log2(len(C))) + 1):
            subset_size = 2 ** i
            if sum(sorted_C[:subset_size]) >= N:
                return True
        return False


    def shuffle_with_indices(self, lst):
        indexed_lst = list(enumerate(lst))
        random.shuffle(indexed_lst)
        shuffled_values = [x[1] for x in indexed_lst]
        original_indices = [x[0] for x in indexed_lst]

        return shuffled_values, original_indices


