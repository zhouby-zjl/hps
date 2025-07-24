import random
import numpy as np

class RTTSimulator:
    def __init__(self, packet_size_bytes, link_bandwidth_gbps, distance_per_link_km, propagation_speed_mps,
                 arrival_rate_pps, service_rate_pps):
        self.packet_size_bits = packet_size_bytes * 8
        self.link_bandwidth_bps = link_bandwidth_gbps * 1e9
        self.distance_per_link_meters = distance_per_link_km * 1000
        self.propagation_speed_mps = propagation_speed_mps
        self.arrival_rate = arrival_rate_pps
        self.service_rate = service_rate_pps

    def transmission_delay(self):
        return self.packet_size_bits / self.link_bandwidth_bps

    def propagation_delay(self):
        return self.distance_per_link_meters / self.propagation_speed_mps

    def queueing_delay(self):
        if self.service_rate > self.arrival_rate:
            rho = self.arrival_rate / self.service_rate
            queueing_delay = rho / (self.service_rate * (1 - rho))
        else:
            queueing_delay = float('inf')
        return queueing_delay

    def processing_delay(self):
        return random.uniform(0.000001, 0.00001)

    def total_delay(self):
        t_tx = self.transmission_delay()
        t_prop = self.propagation_delay()
        t_queue = self.queueing_delay()
        t_proc = self.processing_delay()

        return t_tx + t_prop + t_queue + t_proc

    def compute_rtt(self, len_links):
        one_way_delay = sum(self.total_delay() for _ in range(len_links))
        rtt = 2 * one_way_delay
        return rtt


MTU = 1500
link_bandwidth_gbps = 10
service_rate_pps = int(np.round(link_bandwidth_gbps * 1e9 / (8 * MTU)))
arrival_rate_pps = 500

rtt_simulator = RTTSimulator(
    packet_size_bytes=MTU,  # 1500 bytes, common MTU size
    link_bandwidth_gbps=link_bandwidth_gbps,  # 10 Gbps
    distance_per_link_km=0.05,  # 50m per link, typical for intra-data center links
    propagation_speed_mps=2e8,  # 200,000,000 m/s, close to the speed of light in fiber
    arrival_rate_pps=arrival_rate_pps,  # Arrival rate in packets per second
    service_rate_pps=service_rate_pps  # Service rate in packets per second
)

