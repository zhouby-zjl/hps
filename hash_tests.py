import random
from dcn_networks import dcn_network, packet
import config

def test_ecmp_hash(net, k=5, nnh=8):
    deltas_can = [1 << i for i in range(config.HEADER_CHANGE_BITS)]
    pkt = packet()
    deltas = random.sample(deltas_can, k)
    pkt_zero = pkt.get_packet_zero()
    sw = net.switches_edge[0]
    hv_zero = sw.ecmp_hash_by_ba(pkt_zero, nnh)
    pkt.port_src = 0
    hv_h = sw.ecmp_hash_by_ba(pkt.to_seq(), nnh)
    delta_xored = 0
    o_deltas = []
    o_deltas_xored = 0
    for delta in deltas:
        o_delta = sw.ecmp_hash_by_ba(pkt.get_delta(delta), nnh) ^ hv_zero
        o_deltas.append(o_delta)
        o_deltas_xored ^= o_delta
        delta_xored ^= delta

    pkt.port_src ^= delta_xored
    hv_ecmp = sw.ecmp_hash_by_ba(pkt.to_seq(), nnh)
    hv_expected = hv_h ^ o_deltas_xored
    print(f"hv_ecmp: {hv_ecmp}, hv_expected: {hv_expected}, hv_h: {hv_h}, o_deltas_xored: {o_deltas_xored}")

def test_hashes(net : dcn_network):
    sw0 = net.switches_edge[0]
    sw1 = net.switches_edge[1]
    nnh = 10
    pkt = packet()
    hv0 = sw0.ecmp_hash_by_ba(pkt.to_seq(), nnh)
    hv1 = sw1.ecmp_hash_by_ba(pkt.to_seq(), nnh)
    print(f"hv0: {hv0}, hv1: {hv1}")

def test_hash_collision(net : dcn_network, n_tests):
    net.build_topo()
    sw0 = net.switches_edge[0]
    nnh = 4
    pkt = packet()

    hv_zero = sw0.ecmp_hash_by_ba(pkt.get_packet_zero(), nnh)
    hv_map = {}

    for val in range(2**16):
        hv0 = sw0.ecmp_hash_by_ba(pkt.get_delta(val), nnh)
        if hv0 not in hv_map:
            hv_map[hv0] = []
        hv_map[hv0].append(val)

    hvs = list(hv_map.keys())
    for hv in hvs:
        if len(hv_map[hv]) >= 3:
            val_0 = hv_map[hv][0]
            val_1 = hv_map[hv][1]
            val_2 = hv_map[hv][2]
            val_012 = val_0 ^ val_1 ^ val_2
            delta_012 = pkt.get_delta(val_012)
            hv012 = sw0.ecmp_hash_by_ba(delta_012, nnh)
            right = hv ^ hv ^ hv
            print(f"The same hv: {hv012} =?= {right}")

    if len(hvs) >= 2:
        hv0 = hvs[0]
        hv1 = hvs[1]
        val_0 = hv_map[hv0][0]
        val_1 = hv_map[hv1][0]
        val_01 = val_0 ^ val_1
        delta_01 = pkt.get_delta(val_01)
        hv01 = sw0.ecmp_hash_by_ba(delta_01, nnh)
        right = hv0 ^ hv1 ^ hv_zero
        print(f"The different hv: {hv01} =?= {right}")
