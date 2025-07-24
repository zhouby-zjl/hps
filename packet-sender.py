from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import packet, ethernet, ipv4, udp
from ryu.ofproto import ofproto_v1_3

class SimpleSwitch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch, self).__init__(*args, **kwargs)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.send_custom_udp_packet(datapath)

    def send_custom_udp_packet(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Build Ethernet, IPv4, and UDP headers
        eth = ethernet.ethernet(dst='ff:ff:ff:ff:ff:ff',
                                src='aa:aa:aa:aa:aa:aa',
                                ethertype=0x0800)
        ip = ipv4.ipv4(dst='192.168.1.2',
                       src='192.168.1.1',
                       proto=17)
        udp_pkt = udp.udp(dst_port=5000,
                          src_port=5001)
        p = packet.Packet()
        p.add_protocol(eth)
        p.add_protocol(ip)
        p.add_protocol(udp_pkt)
        p.add_protocol('Custom UDP payload')

        p.serialize()

        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  in_port=ofproto.OFPP_CONTROLLER,
                                  actions=actions,
                                  data=p.data)
        datapath.send_msg(out)
