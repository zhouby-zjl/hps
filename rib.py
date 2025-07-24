from typing import List
from utils import utils
import numpy as np

class route_table:
    routes_hash = None  # routes_hash[dst][port]=metric

    def __init__(self):
        self.routes_hash = {}

    def add_route(self, ip_dst_str, port, metric):
        dst = utils.getIpIn32bits(ip_dst_str)
        dst_exists = dst in self.routes_hash
        if dst_exists and port in self.routes_hash[dst]:
            return
        if not dst_exists:
            self.routes_hash[dst] = {}
        self.routes_hash[dst][port] = metric

    def match_route(self, ip_dst_str):
        dst = utils.getIpIn32bits(ip_dst_str)
        if dst not in self.routes_hash:
            return []
        m = list(self.routes_hash[dst].items())
        return m

    def get_all_dsts(self):
        return set([utils.getIpInStr(dst) for dst in self.routes_hash.keys()])

    def get_size(self):
        return int(np.sum([len(self.routes_hash[dst]) for dst in self.routes_hash]))

    def __repr__(self):
        routes_str = []
        for dst in self.routes_hash:
            routes_str.append(utils.getIpInStr(dst))
            routes_str.append(": ")
            for port, metric in self.routes_hash[dst].items():
                routes_str.append(f"({port}, {metric}) ")
        return "".join(routes_str)


class port_item:
    def __init__(self, id, addr):
        self.id = id
        self.addr = addr

    def __repr__(self):
        return f"port(id={self.id}, addr={self.addr})"

class port_table:
    ports_info : List[port_item] = None
    port_id = 0

    def __init__(self):
        self.port_id = 0
        self.ports_info = []

    def __repr__(self):
        ports_str = []
        for item in self.ports_info:
            ports_str.append(str(item))
        return "\n".join(ports_str)

    def add_port(self, addr):
        self.ports_info.append(port_item(id=self.port_id, addr=addr))
        self.port_id += 1
        return self.port_id - 1