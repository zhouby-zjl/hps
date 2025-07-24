#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/crc32.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <map>
#include "ns3/tcp-dctcp.h"
#include "ns3/traffic-control-module.h"

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE("DcnSim");

#define SEGMENT_SIZE 1448
#define PORT_SRC_ORIGNAL 	0
#define PORT_SRC_VALID 		1

struct Edge {
	uint32_t src, dst;
	string ip1, ip2;
};

struct TraversedPaths {
	vector<uint32_t> traversedNodes;
	vector<uint32_t> ecmpVals;
	vector<uint32_t> outInterfaces;
};


struct IntfInfo {
	uint32_t nodeID;
	uint32_t portID;
	uint32_t intfID;
};

struct FlowPathToScheduleInfo {
	uint32_t 				idx;
	uint32_t 				node_src_id;
	uint32_t 				node_dst_id;
	uint32_t 				port_src;
	uint32_t 				port_dst;
	uint32_t 				valid_port_src;
	double 					event_time;
	int 					flow_size;
	std::vector<uint32_t> 	path_ecmp;
	std::vector<uint32_t> 	designated_path;
	std::vector<uint32_t> 	path_with_valid_port_src;
};

struct SocketState {
	uint32_t totalTxBytes;
	uint32_t currentTxBytes;
	uint8_t data[SEGMENT_SIZE];
};

struct FlowPerf {
	ns3::Time	flowStartTime;
	ns3::Time 	flowCompletedTime;
	uint32_t 	flowSize;
	uint32_t	packetsSent;
	uint32_t	packetsRecv;
	vector<pair<ns3::Time, uint32_t>>
	srcCWndChanges;
	uint32_t 	srcTx, srcRx;
	uint32_t 	dstTx, dstRx;
};


struct SimStates {
	vector<FlowPathToScheduleInfo>* flowPathToScheduleInfo;
	NodeContainer* nodes;
	map<uint32_t, Ipv4Address>* nodeIpMap;
	string flowType;
	uint8_t portSrcType;
};

SimStates* g_simStates = NULL;

uint32_t nodeCount;
unordered_set<uint32_t> epNodes;
vector<Edge> edges;

typedef map<uint32_t, map<uint32_t, pair<uint32_t, Ptr<NetDevice>>>> PortsMap;
PortsMap portsMap;

map<string, IntfInfo> ipToIntfInfoMap;
map<pair<uint32_t, uint32_t>, uint32_t> portAssignments;

uint32_t flowToLoadCursor = 0;
uint32_t N_FLOWS_LOOKAHEAD = 100;

map<uint32_t, uint32_t> g_dataIdSentMap;
map<uint32_t, uint32_t> g_dataIdRecvMap;
map<uint32_t, SocketState> g_sockStates;

map<uint32_t, FlowPerf> g_flowToPerfInfo;
std::map<uint64_t, uint32_t> g_deviceDropCounts;
std::map<uint64_t, uint32_t> g_deviceSentCounts;
uint32_t g_totalFlows = 0;
uint32_t g_totalFlowsLoaded = 0;
uint32_t g_flowFinished = 0;
uint32_t g_flowStarted = 0;

double g_totalSimulationTimeInSecs = 10.0;

void FlowsLoaderLazy();

string getIPStr(Ipv4Address addr) {
	stringstream ss;
	ss << addr;
	return ss.str();
}

Ipv4Address ToIpv4Address(const string& ipStr) {
	return Ipv4Address(ipStr.c_str());
}

string getVectorStr(vector<uint32_t> vec) {
	uint32_t i = 0, n = vec.size();
	stringstream ss;
	for (uint32_t x : vec) {
		ss << x << (i != n - 1 ? ", " : "");
		++i;
	}
	return ss.str();
}

std::vector<uint32_t> ParsePathString(const std::string& pathStr) {
	std::vector<uint32_t> result;
	std::stringstream ss(pathStr);
	std::string item;
	while (std::getline(ss, item, '-')) {
		result.push_back(static_cast<uint32_t>(std::stoi(item)));
	}
	return result;
}


void ParseTopology(const string &filename,
		uint32_t &nodeCount,
		unordered_set<uint32_t> &epNodes,
		vector<Edge> &edges) {
	ifstream infile(filename);
	string line;
	enum Section { NONE, NODES, EP_IDS, EDGES } section = NONE;

	while (getline(infile, line)) {
		if (line == "nodes:") {
			section = NODES;
		} else if (line == "ep_ids:") {
			section = EP_IDS;
		} else if (line == "edges:") {
			section = EDGES;
		} else if (!line.empty()) {
			stringstream ss(line);
			if (section == NODES) {
				ss >> nodeCount;
			} else if (section == EP_IDS) {
				string idStr;
				while (getline(ss, idStr, ',')) {
					epNodes.insert(stoi(idStr));
				}
			} else if (section == EDGES) {
				Edge edge;
				string tmp;
				getline(ss, tmp, ',');
				edge.src = stoi(tmp);
				getline(ss, tmp, ',');
				edge.dst = stoi(tmp);
				getline(ss, edge.ip1, ',');
				getline(ss, edge.ip2);
				edges.push_back(edge);
			}
		}
	}
}

std::vector<FlowPathToScheduleInfo> LoadFlowPathToScheduleFromFiles(
    const std::string& flowPathToScheduleInfoFile) {

    std::vector<FlowPathToScheduleInfo> flowPathToScheduleInfos;

    std::ifstream file(flowPathToScheduleInfoFile);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << flowPathToScheduleInfoFile << std::endl;
        return flowPathToScheduleInfos;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        FlowPathToScheduleInfo info;

        // Read 7 numeric fields
        std::getline(ss, token, ','); info.idx = std::stoul(token);
        std::getline(ss, token, ','); info.node_src_id = std::stoul(token);
        std::getline(ss, token, ','); info.node_dst_id = std::stoul(token);
        std::getline(ss, token, ','); info.port_src = std::stoul(token);
        std::getline(ss, token, ','); info.port_dst = std::stoul(token);
        std::getline(ss, token, ','); info.valid_port_src = std::stoul(token);
        std::getline(ss, token, ','); info.event_time = std::stod(token);
        std::getline(ss, token, ','); info.flow_size = std::stoul(token);

        // Read 3 path strings and convert to vectors
        std::getline(ss, token, ','); info.path_ecmp = ParsePathString(token);
        std::getline(ss, token, ','); info.designated_path = ParsePathString(token);
        std::getline(ss, token, ','); info.path_with_valid_port_src = ParsePathString(token);

        flowPathToScheduleInfos.push_back(info);
    }

    file.close();

    return flowPathToScheduleInfos;
}



void ParsePortMap(const string &filename,
		map<std::pair<uint32_t, uint32_t>, uint32_t> &portMap) {
	ifstream infile(filename);
	string line;
	while (getline(infile, line)) {
		stringstream ss(line);
		string s;
		uint32_t a, b, port;
		getline(ss, s, ','); a = stoi(s);
		getline(ss, s, ','); b = stoi(s);
		getline(ss, s);     port = stoi(s);
		portMap[{a, b}] = port;
	}
}

void CheckPortAssignmentsCoverage(
		const vector<Edge> &edges,
		const map<std::pair<uint32_t, uint32_t>, uint32_t> &portAssignments) {
	for (const auto &edge : edges) {
		auto fwd = portAssignments.find({edge.src, edge.dst});
		auto rev = portAssignments.find({edge.dst, edge.src});
		if (fwd == portAssignments.end()) {
			NS_FATAL_ERROR("Missing port assignment for edge: " << edge.src << " -> " << edge.dst);
		}
		if (rev == portAssignments.end()) {
			NS_FATAL_ERROR("Missing port assignment for edge: " << edge.dst << " -> " << edge.src);
		}
	}
}



void WriteFlowPerfToCsv(const std::string& filename) {
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening file " << filename << std::endl;
		return;
	}

	// Write CSV header
	file << "FlowId,FlowStartTime,FlowCompletedTime,FlowSize,PacketsSent,PacketsRecv,SrcCWndChanges,SrcTx,SrcRx,DstTx,DstRx\n";

	for (const auto& entry : g_flowToPerfInfo) {
		uint32_t flowId = entry.first;
		const FlowPerf& perf = entry.second;

		file << flowId << ","
				<< perf.flowStartTime.GetSeconds() << ","
				<< perf.flowCompletedTime.GetSeconds() << ","
				<< perf.flowSize << ","
				<< perf.packetsSent << ","
				<< perf.packetsRecv << ",";

		// Serialize srcCWndChanges
		std::ostringstream cwndStream;
		for (size_t i = 0; i < perf.srcCWndChanges.size(); ++i) {
			cwndStream << perf.srcCWndChanges[i].first.GetSeconds() << "~"
					<< perf.srcCWndChanges[i].second;
			if (i != perf.srcCWndChanges.size() - 1) {
				cwndStream << "#";
			}
		}
		file << "\"" << cwndStream.str() << "\",";  // Put in quotes to avoid CSV parsing issues

		file << perf.srcTx << ","
				<< perf.srcRx << ","
				<< perf.dstTx << ","
				<< perf.dstRx << "\n";
	}

	file.close();
	std::cout << "Flow performance metrics written to " << filename << std::endl;
}



class MyIpv4EcmpRouting : public Ipv4StaticRouting
{
public:
	struct PacketInfo {
		uint16_t id;
		Ipv4Address srcIp, dstIp;
		uint16_t srcPort, dstPort;
		uint8_t proto; // 0 for TCP, 1 for UDP
		uint32_t dataId;
		bool toTrack;
	};

	static TypeId GetTypeId (void);
	MyIpv4EcmpRouting ();
	~MyIpv4EcmpRouting ();

	void SetPermutation (const vector<uint32_t>& perm);
	void SetSeed (const vector<uint8_t>& seed);
	void SetRoutingTable (const map<Ipv4Address, vector<uint32_t>>& table);
	void SetNodeID(uint32_t nodeID);
	PacketInfo GetPacketInfo(Ptr<const Packet> p, const Ipv4Header &header);
	void trackPacketPath(PacketInfo info, uint32_t curNodeId, uint32_t ecmpVal, uint32_t outInterface);

	// Ipv4RoutingProtocol interface
	Ptr<Ipv4Route> RouteOutput (Ptr<Packet> p, const Ipv4Header &header,
			Ptr<NetDevice> oif, Socket::SocketErrno &sockerr) override;
	bool RouteInput (Ptr<const Packet> p, const Ipv4Header &header, Ptr<const NetDevice> idev,
			UnicastForwardCallback ucb, MulticastForwardCallback mcb,
			LocalDeliverCallback lcb, ErrorCallback ecb) override;

	void NotifyInterfaceUp (uint32_t interface) override {}
	void NotifyInterfaceDown (uint32_t interface) override {}
	void NotifyAddAddress (uint32_t interface, Ipv4InterfaceAddress address) override {}
	void NotifyRemoveAddress (uint32_t interface, Ipv4InterfaceAddress address) override {}
	void SetIpv4 (Ptr<Ipv4> ipv4) override { m_ipv4 = ipv4; }
	void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit) const override {}

	static void printTrackedPaths();

	static map<uint32_t, map<uint32_t, map<uint32_t, TraversedPaths>>> g_pairToTraversedPathMap;

private:
	int32_t ComputeEcmpHash (PacketInfo info, uint32_t nNexthop);

	vector<uint32_t> m_permutation;
	vector<uint8_t> m_seed;
	map<Ipv4Address, vector<uint32_t>> m_routingTable;

	Ptr<Ipv4> m_ipv4;
	uint32_t nodeID;
};

map<uint32_t, map<uint32_t, map<uint32_t, TraversedPaths>>> MyIpv4EcmpRouting::g_pairToTraversedPathMap;

TypeId
MyIpv4EcmpRouting::GetTypeId (void)
{
	static TypeId tid = TypeId ("ns3::MyIpv4EcmpRouting")
    						.SetParent<Ipv4StaticRouting> ()
							.SetGroupName ("Internet")
							.AddConstructor<MyIpv4EcmpRouting> ();
	return tid;
}

MyIpv4EcmpRouting::MyIpv4EcmpRouting () {
}

void MyIpv4EcmpRouting::SetNodeID(uint32_t nodeID) {
	this->nodeID = nodeID;
}

MyIpv4EcmpRouting::~MyIpv4EcmpRouting () {}

void MyIpv4EcmpRouting::SetPermutation (const vector<uint32_t>& perm) {
	m_permutation = perm;
}

void MyIpv4EcmpRouting::SetSeed (const vector<uint8_t>& seed) {
	m_seed = seed;
}

void MyIpv4EcmpRouting::SetRoutingTable (
		const map<Ipv4Address, vector<uint32_t>>& table) {
	m_routingTable = table;
}

MyIpv4EcmpRouting::PacketInfo MyIpv4EcmpRouting::GetPacketInfo(Ptr<const Packet> p, const Ipv4Header &header) {
	PacketInfo info;
	info.id = 0;
	info.srcPort = 0;
	info.dstPort = 0;
	info.proto = 0xff;
	info.toTrack = false;

	if (p == NULL) {
		return info;
	}

	Ptr<Packet> pktCopy = p->Copy();

	uint8_t protocol = header.GetProtocol();
	info.id = header.GetIdentification();
	info.srcIp = header.GetSource();
	info.dstIp = header.GetDestination();

	if (protocol == 6) // TCP
	{
		TcpHeader tcpHeader;
		pktCopy->RemoveHeader(tcpHeader);
		info.srcPort = tcpHeader.GetSourcePort();
		info.dstPort = tcpHeader.GetDestinationPort();
		info.proto = 0;
	}
	else if (protocol == 17) // UDP
	{
		UdpHeader udpHeader;
		pktCopy->RemoveHeader(udpHeader);
		info.srcPort = udpHeader.GetSourcePort();
		info.dstPort = udpHeader.GetDestinationPort();
		info.proto = 1;
	}
	else {
		return info;
	}

	uint8_t buffer[4];
	if (pktCopy->GetSize() >= 4)
	{
		pktCopy->CopyData(buffer, 4);
		uint32_t value = (buffer[0] << 24) |
				(buffer[1] << 16) |
				(buffer[2] << 8) |
				(buffer[3]);

		info.dataId = value & 0x7fffffff;
		info.toTrack = (value >> 31) == 1;
	}

	return info;
}

void MyIpv4EcmpRouting::trackPacketPath(PacketInfo info, uint32_t curNodeId, uint32_t ecmpVal, uint32_t outInterface) {
	if (info.toTrack) {
		string srcIpStr = getIPStr(info.srcIp);
		string dstIpStr = getIPStr(info.dstIp);
		uint32_t srcNodeId = 0xffffffff, dstNodeId = 0xffffffff;
		if (ipToIntfInfoMap.find(srcIpStr) != ipToIntfInfoMap.end()) {
			srcNodeId = ipToIntfInfoMap[srcIpStr].nodeID;
		}
		if (ipToIntfInfoMap.find(dstIpStr) != ipToIntfInfoMap.end()) {
			dstNodeId = ipToIntfInfoMap[dstIpStr].nodeID;
		}
		if (srcNodeId == 0xffffffff || dstNodeId == 0xffffffff) return;

		if (g_pairToTraversedPathMap.find(srcNodeId) == g_pairToTraversedPathMap.end()) {
			g_pairToTraversedPathMap[srcNodeId];
		}
		if (g_pairToTraversedPathMap[srcNodeId].find(dstNodeId) == g_pairToTraversedPathMap[srcNodeId].end()) {
			g_pairToTraversedPathMap[srcNodeId][dstNodeId];
		}
		if (g_pairToTraversedPathMap[srcNodeId][dstNodeId].find(info.id) ==
				g_pairToTraversedPathMap[srcNodeId][dstNodeId].end()) {
			g_pairToTraversedPathMap[srcNodeId][dstNodeId][info.id];
		}

		TraversedPaths& path = g_pairToTraversedPathMap[srcNodeId][dstNodeId][info.id];
		path.traversedNodes.push_back(curNodeId);
		path.ecmpVals.push_back(ecmpVal);
		path.outInterfaces.push_back(outInterface);
	}
}

Ptr<Ipv4Route>
MyIpv4EcmpRouting::RouteOutput (Ptr<Packet> p, const Ipv4Header &header,
		Ptr<NetDevice> oif, Socket::SocketErrno &sockerr)
{
	uint32_t nRoutes = this->GetNRoutes();
	vector<Ipv4RoutingTableEntry> route_matched;
	for (uint32_t j = 0; j < nRoutes; ++j) {
		Ipv4RoutingTableEntry route = this->GetRoute(j);
		if (route.GetDest() == header.GetDestination()) {
			route_matched.push_back(route);
		}
	}

	if (route_matched.size() == 0) {
		sockerr = Socket::ERROR_NOROUTETOHOST;
		return nullptr;
	}

	int32_t hv = 0;
	PacketInfo info;
	if (p != NULL) {
		info = this->GetPacketInfo(p, header);
		hv = ComputeEcmpHash (info, route_matched.size ());
		if (hv < 0 || static_cast<size_t>(hv) >= route_matched.size ()) {
			sockerr = Socket::ERROR_NOROUTETOHOST;
			return nullptr;
		}
	}

	uint32_t outInterface = route_matched[hv].GetInterface();

	Ptr<Ipv4Route> route = Create<Ipv4Route> ();
	route->SetDestination (header.GetDestination());
	route->SetGateway (route_matched[hv].GetGateway());
	route->SetOutputDevice (m_ipv4->GetNetDevice (outInterface));
	route->SetSource (m_ipv4->GetAddress (outInterface, 0).GetLocal ());
	sockerr = Socket::ERROR_NOTERROR;

	// cout << "FWD (" << this->nodeID << ") @ " << Simulator::Now() << ". OutIntf: " <<  outInterface << endl;

	if (p != NULL) {
		this->trackPacketPath(info, this->nodeID, hv, outInterface);
	}

	return route;
}

bool
MyIpv4EcmpRouting::RouteInput(Ptr<const Packet> p, const Ipv4Header &header,
		Ptr<const NetDevice> idev,
		UnicastForwardCallback ucb,
		MulticastForwardCallback mcb,
		LocalDeliverCallback lcb,
		ErrorCallback ecb)
{
	Ipv4Address dst = header.GetDestination();

	int32_t iif = m_ipv4->GetInterfaceForDevice(idev);
	NS_ASSERT(iif >= 0);

	if (m_ipv4->IsDestinationAddress(dst, iif)) {
		if (!lcb.IsNull()) {
			// NS_LOG_INFO("Local delivery at node " << nodeID << " to " << dst);
			lcb(p, header, iif);
			return true;
		} else {
			NS_LOG_WARN("Local delivery callback is null at node " << nodeID);
			return false;
		}
	}

	// Check if the node is configured to forward packets
	if (!m_ipv4->IsForwarding(iif)) {
		NS_LOG_WARN("Forwarding disabled at node " << nodeID << " on interface " << iif);
		ecb(p, header, Socket::ERROR_NOROUTETOHOST);
		return true;
	}

	// Match routes with exact destination
	uint32_t nRoutes = this->GetNRoutes();
	std::vector<Ipv4RoutingTableEntry> route_matched;
	for (uint32_t j = 0; j < nRoutes; ++j) {
		const Ipv4RoutingTableEntry &route = this->GetRoute(j);
		if (route.GetDest() == dst) {
			route_matched.push_back(route);
		}
	}

	if (route_matched.empty()) {
		NS_LOG_WARN("No route to destination " << dst << " at node " << nodeID);
		ecb(p, header, Socket::ERROR_NOROUTETOHOST);
		return false;
	}

	PacketInfo info;
	int32_t hv = 0;
	if (p != NULL) {
		// Select one route using ECMP hash
		info = this->GetPacketInfo(p, header);

		hv = ComputeEcmpHash(info, route_matched.size());
		if (hv < 0 || static_cast<size_t>(hv) >= route_matched.size()) {
			NS_LOG_ERROR("Invalid ECMP hash result " << hv << " at node " << nodeID);
			ecb(p, header, Socket::ERROR_NOROUTETOHOST);
			return false;
		}
	}

	const Ipv4RoutingTableEntry &selectedRoute = route_matched[hv];
	uint32_t outInterface = selectedRoute.GetInterface();

	Ptr<Ipv4Route> route = Create<Ipv4Route>();
	route->SetDestination(dst);
	route->SetGateway(selectedRoute.GetGateway());
	route->SetOutputDevice(m_ipv4->GetNetDevice(outInterface));
	route->SetSource(m_ipv4->GetAddress(selectedRoute.GetInterface(), 0).GetLocal());

	// Forward using unicast callback
	ucb(route, p, header);

	// NS_LOG_INFO("Forwarding at node " << nodeID << " to " << info.dstIp << " via interface " << outInterface);
	// std::cout << "FWD (" << nodeID << ") @ " << Simulator::Now() << ". OutIntf: " << selectedRoute.GetInterface() << std::endl;

	if (p != NULL) {
		this->trackPacketPath(info, this->nodeID, hv, outInterface);
	}
	return true;
}


int32_t
MyIpv4EcmpRouting::ComputeEcmpHash (MyIpv4EcmpRouting::PacketInfo info, uint32_t nNexthop)
{
	if (nNexthop == 1) {
		return 0;
	}

	if (nNexthop == 0 || m_permutation.empty () || m_seed.empty ()) return -1;

	// Create bytearray: [ip_src][ip_dst][port_src][port_dst]
	vector<uint8_t> ba;
	for (int i = 0; i < 4; ++i)
	{
		ba.push_back (info.srcIp.Get () >> (24 - 8 * i) & 0xFF);
	}
	for (int i = 0; i < 4; ++i)
	{
		ba.push_back (info.dstIp.Get () >> (24 - 8 * i) & 0xFF);
	}
	ba.push_back ((info.srcPort >> 8) & 0xFF);
	ba.push_back (info.srcPort & 0xFF);
	ba.push_back ((info.dstPort >> 8) & 0xFF);
	ba.push_back (info.dstPort & 0xFF);

	// Apply permutation
	vector<uint8_t> ba_perm (m_permutation.size ());
	for (size_t i = 0; i < m_permutation.size (); ++i)
	{
		ba_perm[i] = ba[m_permutation[i]];
	}

	// XOR with seed
	vector<uint8_t> xored;
	for (size_t i = 0; i < ba_perm.size (); ++i)
	{
		xored.push_back (ba_perm[i] ^ m_seed[i]);
	}

	// CRC32
	uint32_t h = CRC32Calculate (xored.data (), xored.size ());
	int32_t hv = h % nNexthop;

	return hv;
}

void MyIpv4EcmpRouting::printTrackedPaths() {
	for (const auto& srcPair : g_pairToTraversedPathMap) {
		uint32_t srcNodeId = srcPair.first;
		const auto& dstMap = srcPair.second;
		for (const auto& dstPair : dstMap) {
			uint32_t dstNodeId = dstPair.first;
			const auto& dataMap = dstPair.second;
			std::cout << "PAIR " << srcNodeId << "->" << dstNodeId << ":" << std::endl;
			for (const auto& dataPair : dataMap) {
				uint32_t dataId = dataPair.first;
				const TraversedPaths& path = dataPair.second;
				std::cout << "path for data ID " << dataId << ": ("
						<< getVectorStr(path.traversedNodes)
						<< "), ecmp vals (" << getVectorStr(path.ecmpVals)
						<< "), out interfaces (" << getVectorStr(path.outInterfaces)
						<< ")" << std::endl;
			}
		}
	}
}


class MyIpv4EcmpRoutingHelper : public Ipv4StaticRoutingHelper {
public:
	MyIpv4EcmpRoutingHelper() {}
	MyIpv4EcmpRoutingHelper(const MyIpv4EcmpRoutingHelper &o) {}

	virtual ~MyIpv4EcmpRoutingHelper() {}

	virtual MyIpv4EcmpRoutingHelper* Copy(void) const {
		return new MyIpv4EcmpRoutingHelper(*this);
	}

	virtual Ptr<Ipv4RoutingProtocol> Create(Ptr<Node> node) const {
		Ptr<MyIpv4EcmpRouting> routing = CreateObject<MyIpv4EcmpRouting>();
		uint32_t nodeID = node->GetId();
		routing->SetNodeID(nodeID);
		node->AggregateObject(routing);
		return routing;
	}
};


void ConfigureSeedsAndPermutations(const string& seedsFile,
		const string& permFile,
		const NodeContainer& nodes) {
	ifstream seedsIn(seedsFile);
	ifstream permIn(permFile);
	string line;

	map<uint32_t, vector<uint8_t>> seedsMap;
	map<uint32_t, vector<uint32_t>> permMap;

	// Parse seeds
	while (getline(seedsIn, line)) {
		stringstream ss(line);
		string idStr, valuesStr;
		if (getline(ss, idStr, ':') && getline(ss, valuesStr)) {
			uint32_t nodeId = stoul(idStr);
			vector<uint8_t> seed;
			stringstream vs(valuesStr);
			string num;
			while (getline(vs, num, ',')) {
				seed.push_back(static_cast<uint8_t>(stoi(num)));
			}
			seedsMap[nodeId] = seed;
		}
	}

	// Parse permutations
	while (getline(permIn, line)) {
		stringstream ss(line);
		string idStr, valuesStr;
		if (getline(ss, idStr, ':') && getline(ss, valuesStr)) {
			uint32_t nodeId = stoul(idStr);
			vector<uint32_t> perm;
			stringstream vs(valuesStr);
			string num;
			while (getline(vs, num, ',')) {
				perm.push_back(static_cast<uint32_t>(stoi(num)));
			}
			permMap[nodeId] = perm;
		}
	}

	// Assign seed and permutation to corresponding nodes
	for (uint32_t i = 0; i < nodes.GetN(); ++i) {
		Ptr<Node> node = nodes.Get(i);
		Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
		Ptr<Ipv4RoutingProtocol> proto = ipv4->GetRoutingProtocol();
		Ptr<MyIpv4EcmpRouting> ecmp = DynamicCast<MyIpv4EcmpRouting>(proto);

		if (ecmp) {
			if (seedsMap.count(i)) {
				ecmp->SetSeed(seedsMap[i]);
			}
			if (permMap.count(i)) {
				ecmp->SetPermutation(permMap[i]);
			}
		} else {
			NS_LOG_WARN("Node " << i << " does not have MyIpv4EcmpRouting");
		}
	}
}



void ConfigureStaticRoutesFromFile(const string& filename, NodeContainer& allNodes, PortsMap& portsMap) {
	LogComponentEnable("DcnSim", LOG_LEVEL_INFO);

	ifstream infile(filename);
	if (!infile.is_open()) {
		NS_LOG_ERROR("Failed to open file: " << filename);
		return;
	}

	string line;
	while (getline(infile, line)) {
		if (line.empty()) continue;

		// Parse node ID
		stringstream ss(line);
		string nodeIdStr;
		if (!getline(ss, nodeIdStr, ':')) continue;
		uint32_t nodeId = stoul(nodeIdStr);

		Ptr<Node> node = allNodes.Get(nodeId);
		Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
		MyIpv4EcmpRoutingHelper routingHelper;
		Ptr<Ipv4StaticRouting> staticRouting_ = routingHelper.GetStaticRouting(ipv4);
		Ptr<MyIpv4EcmpRouting> staticRouting = DynamicCast<MyIpv4EcmpRouting>(staticRouting_);

		string routesStr;
		if (!getline(ss, routesStr)) continue;

		stringstream routesStream(routesStr);
		string routeEntry;
		stringstream ss2;
		while (getline(routesStream, routeEntry, '|')) {
			size_t commaPos = routeEntry.find(',');
			if (commaPos == string::npos) continue;

			string destIpStr = routeEntry.substr(0, commaPos);
			string portListStr = routeEntry.substr(commaPos + 1);
			Ipv4Address destIp(destIpStr.c_str());

			stringstream portStream(portListStr);
			string portIndexStr;
			vector<uint32_t> portIndices;
			while (getline(portStream, portIndexStr, '-')) {
				portIndices.push_back(stoul(portIndexStr));
			}

			uint32_t currentNodeId = nodeId;
			Ptr<NetDevice> outDev = nullptr;
			Ptr<NetDevice> peerDev = nullptr;

			// Traverse multi-hop ports
			for (uint32_t portIndex : portIndices) {
				Ptr<Channel> channel = NULL;
				for (auto iter = portsMap[currentNodeId].begin(); iter != portsMap[currentNodeId].end(); ++iter) {
					if (iter->second.first == portIndex) {
						outDev = iter->second.second;
						channel = outDev->GetChannel();
						break;
					}
				}

				if (channel == nullptr || channel->GetNDevices() < 2) break;

				// Get peer device
				peerDev = (channel->GetDevice(0) == outDev) ?
						channel->GetDevice(1) : channel->GetDevice(0);

				if (outDev == nullptr || peerDev == nullptr) continue;

				// Get the next-hop IP address from the peer device
				Ptr<Node> nextHopNode = peerDev->GetNode();
				Ptr<Ipv4> nextHopIpv4 = nextHopNode->GetObject<Ipv4>();

				// Find the interface index of peerDev
				int32_t interfaceIndex = nextHopIpv4->GetInterfaceForDevice(peerDev);
				if (interfaceIndex == -1) continue;

				Ipv4Address nextHopIp = nextHopIpv4->GetAddress(interfaceIndex, 0).GetLocal();

				ss2.str("");
				ss2 << nextHopIp;
				string nextHopIpStr = ss2.str();

				// Install host route
				staticRouting->AddHostRouteTo(destIp, nextHopIp, outDev->GetIfIndex());
			}
		}
	}

	infile.close();
}

void DumpAllRoutingTables(const NodeContainer &nodes) {
	MyIpv4EcmpRoutingHelper routingHelper;
	cout << "\n========== ROUTING TABLES ==========" << endl;

	for (uint32_t i = 0; i < nodes.GetN(); ++i) {
		Ptr<Node> node = nodes.Get(i);
		Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
		Ptr<MyIpv4EcmpRouting> staticRouting = DynamicCast<MyIpv4EcmpRouting>(routingHelper.GetStaticRouting(ipv4));

		cout << "\nRouting Table of Node " << i << ":" << endl;
		uint32_t nRoutes = staticRouting->GetNRoutes();
		for (uint32_t j = 0; j < nRoutes; ++j) {
			Ipv4RoutingTableEntry route = staticRouting->GetRoute(j);
			cout << "  Destination: " << route.GetDest()
                    						  << ", Mask: " << route.GetDestNetworkMask()
											  << ", Gateway: " << route.GetGateway()
											  << ", Interface: " << route.GetInterface()
											  << ", IsHost: " << (route.IsHost() ? "Yes" : "No") << endl;
		}
	}

	cout << "====================================\n" << endl;
}





void WriteUntilBufferFull (uint32_t nodeId, uint32_t flowId, Ptr<Socket> localSocket, uint32_t txSpace) {
	auto it = g_sockStates.find(flowId);
	if (it == g_sockStates.end()) {
		return; // sock has been closed.
	}


	SocketState& state = g_sockStates[flowId];
	// cout << Simulator::Now() << ", Node " << nodeId << " write: " << state.totalTxBytes << endl;


	while (state.currentTxBytes < state.totalTxBytes && localSocket->GetTxAvailable () > 0) {
		uint32_t left = state.totalTxBytes - state.currentTxBytes;
		uint32_t dataOffset = state.currentTxBytes % SEGMENT_SIZE;
		uint32_t toWrite = SEGMENT_SIZE - dataOffset;
		toWrite = std::min (toWrite, left);
		toWrite = std::min (toWrite, localSocket->GetTxAvailable ());
		int amountSent = localSocket->Send (&state.data[dataOffset], toWrite, 0);
		if(amountSent < 0) {
			return;
		}
		state.currentTxBytes += amountSent;

		g_flowToPerfInfo[flowId].packetsSent += amountSent;
	}

	if (state.currentTxBytes >= state.totalTxBytes) {
		localSocket->Close ();
		g_sockStates.erase(flowId);
	}
}

void StartFlow (uint32_t nodeId, uint32_t flowId, Ptr<Socket> localSocket, Ipv4Address servAddress, uint16_t servPort) {
	g_flowToPerfInfo[flowId].flowStartTime = Simulator::Now();

	localSocket->Connect (InetSocketAddress (servAddress, servPort));

	localSocket->SetSendCallback (MakeBoundCallback (&WriteUntilBufferFull, nodeId, flowId));
	WriteUntilBufferFull (nodeId, flowId, localSocket, localSocket->GetTxAvailable ());

	++g_flowStarted;
}

void ReceiveFlow(uint32_t nodeId, uint32_t flowId, Ptr<Socket> socket) {
	Ptr<Packet> packet;
	uint32_t totalRx = 0, nRecv = 0;
	while ((packet = socket->Recv())) {
		nRecv = packet->GetSize ();
		totalRx += nRecv;

		if (g_flowToPerfInfo.find(flowId) != g_flowToPerfInfo.end()) {
			g_flowToPerfInfo[flowId].packetsRecv += nRecv;
		}

		// cout << Simulator::Now() << ", Node " << nodeId << " recv: " << nRecv << endl;
	}

}

void HandleAcceptedConnectionForFlows(uint32_t nodeId, uint32_t flowId, Ptr<Socket> socket, const Address &from)
{
	socket->SetRecvCallback(MakeBoundCallback(&ReceiveFlow, nodeId, flowId));
}



FlowPathToScheduleInfo* FindFlowPathInfo(std::vector<FlowPathToScheduleInfo>& flowPathInfoVec, uint32_t idx) {
	for (auto& info : flowPathInfoVec) {
		if (info.idx == idx) {
			return &info;
		}
	}
	cout << "No matching FlowPathInfo found for the given idx " << idx << "." << endl;
	return NULL;
}


void CwndChange(uint32_t srcNodeId, uint32_t flowIndex, uint32_t oldCwnd, uint32_t newCwnd) {
	if (g_flowToPerfInfo.find(flowIndex) == g_flowToPerfInfo.end()) return;
	g_flowToPerfInfo[flowIndex].srcCWndChanges.push_back(pair<ns3::Time, uint32_t>(ns3::Now(), newCwnd));
}

void TxChange(uint32_t srcNodeId, uint32_t flowIndex, bool isSrc,
		const Ptr< const Packet > packet, const TcpHeader &header,
		const Ptr< const TcpSocketBase > socket) {
	if (g_flowToPerfInfo.find(flowIndex) == g_flowToPerfInfo.end()) return;
	if (isSrc) {
		g_flowToPerfInfo[flowIndex].srcTx++;
	} else {
		g_flowToPerfInfo[flowIndex].dstTx++;
	}
}

void RxChange(uint32_t srcNodeId, uint32_t flowIndex, bool isSrc,
		const Ptr< const Packet > packet, const TcpHeader &header,
		const Ptr< const TcpSocketBase > socket) {
	if (g_flowToPerfInfo.find(flowIndex) == g_flowToPerfInfo.end()) return;
	if (isSrc) {
		g_flowToPerfInfo[flowIndex].srcRx++;
	} else {
		g_flowToPerfInfo[flowIndex].dstRx++;
	}
}

void StateChange(uint32_t srcNodeId, uint32_t flowIndex,
		const TcpSocket::TcpStates_t oldValue, const TcpSocket::TcpStates_t newValue) {
	if (g_flowToPerfInfo.find(flowIndex) == g_flowToPerfInfo.end()) return;
	cout << "StateChange at node " << srcNodeId << " (" << flowIndex << ") from state " << oldValue << " to state " << newValue << " @ " << Simulator::Now() << endl;
	if (newValue == ns3::TcpSocketBase::FIN_WAIT_2 || newValue == ns3::TcpSocketBase::CLOSED) {
		g_flowToPerfInfo[flowIndex].flowCompletedTime = Simulator::Now();
		++g_flowFinished;
	}
}

void PacketDropPerDevice(uint32_t node_a, uint32_t node_b, Ptr<const Packet> packet) {
	uint64_t node_ab = (uint64_t) node_a << 32 | (uint64_t) node_b;
	g_deviceDropCounts[node_ab]++;
}

void PacketSentPerDevice(uint32_t node_a, uint32_t node_b, Ptr<const Packet> packet) {
	uint64_t node_ab = (uint64_t) node_a << 32 | (uint64_t) node_b;
	g_deviceSentCounts[node_ab]++;
}

void FlowLoader(FlowPathToScheduleInfo& schItem) {
	Ptr<Node> srcNode = g_simStates->nodes->Get(schItem.node_src_id);
	Ptr<Node> dstNode = g_simStates->nodes->Get(schItem.node_dst_id);
	Ipv4Address dstIp = g_simStates->nodeIpMap->at(schItem.node_dst_id);

	FlowPathToScheduleInfo* flowPath = FindFlowPathInfo(*g_simStates->flowPathToScheduleInfo, schItem.idx);

	if (flowPath == NULL) {
		return;
	}

	uint16_t port_src = g_simStates->portSrcType == PORT_SRC_ORIGNAL ? flowPath->port_src : flowPath->valid_port_src;

	// Create sink socket on destination
	Ptr<Socket> sinkSocket, srcSocket;
	if (g_simStates->flowType == "tcp" || g_simStates->flowType == "dctcp") {
		sinkSocket = Socket::CreateSocket(dstNode, TcpSocketFactory::GetTypeId());
		InetSocketAddress localAddr = InetSocketAddress(Ipv4Address::GetAny(), flowPath->port_dst);
		sinkSocket->Bind(localAddr);
		sinkSocket->Listen();
		sinkSocket->SetRecvCallback (MakeBoundCallback (&ReceiveFlow, flowPath->node_dst_id, schItem.idx));
		sinkSocket->SetAcceptCallback (
				MakeNullCallback<bool, Ptr<Socket>, const Address &> (),
				MakeBoundCallback (&HandleAcceptedConnectionForFlows, flowPath->node_dst_id, schItem.idx));

		srcSocket = Socket::CreateSocket(srcNode, TcpSocketFactory::GetTypeId());

		srcSocket->TraceConnectWithoutContext("CongestionWindow",
				MakeBoundCallback(&CwndChange, flowPath->node_src_id, schItem.idx));

		srcSocket->TraceConnectWithoutContext("Tx",
				MakeBoundCallback(&TxChange, flowPath->node_src_id, schItem.idx, true));

		srcSocket->TraceConnectWithoutContext("Rx",
				MakeBoundCallback(&RxChange, flowPath->node_src_id, schItem.idx, true));

		sinkSocket->TraceConnectWithoutContext("Tx",
				MakeBoundCallback(&TxChange, flowPath->node_src_id, schItem.idx, false));

		sinkSocket->TraceConnectWithoutContext("Rx",
				MakeBoundCallback(&RxChange, flowPath->node_src_id, schItem.idx, false));

		srcSocket->TraceConnectWithoutContext("State",
				MakeBoundCallback(&StateChange, flowPath->node_src_id, schItem.idx));

	} else if (g_simStates->flowType == "udp") {
		sinkSocket = Socket::CreateSocket(dstNode, UdpSocketFactory::GetTypeId());
		InetSocketAddress localAddr = InetSocketAddress(Ipv4Address::GetAny(), flowPath->port_dst);
		sinkSocket->Bind(localAddr);
		sinkSocket->SetRecvCallback(MakeBoundCallback(&ReceiveFlow, flowPath->node_dst_id, schItem.idx));

		srcSocket = Socket::CreateSocket(srcNode, UdpSocketFactory::GetTypeId());
	}


	srcSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), port_src));
	// srcSocket->Connect(InetSocketAddress(dstIp, flowPath.port_dst));

	g_dataIdSentMap[flowPath->node_src_id] = 0;
	g_dataIdRecvMap[flowPath->node_dst_id] = -1;


	SocketState state;
	state.totalTxBytes = schItem.flow_size;
	state.currentTxBytes = 0;
	memset(&state.data, 0, SEGMENT_SIZE);
	g_sockStates[schItem.idx] = state;

	FlowPerf perf;
	perf.flowStartTime = Seconds(0);
	perf.flowCompletedTime = Seconds(0);
	perf.flowSize = schItem.flow_size;
	perf.packetsRecv = 0;
	perf.packetsSent = 0;
	perf.srcRx = 0;
	perf.dstRx = 0;
	perf.srcTx = 0;
	perf.dstTx = 0;
	g_flowToPerfInfo[schItem.idx] = perf;

	Simulator::Schedule(Seconds(schItem.event_time), &StartFlow, flowPath->node_src_id, schItem.idx,
			srcSocket, dstIp, flowPath->port_dst);
}


void FlowsLoaderLazy() {
	uint32_t lookaheadFlowIdx = flowToLoadCursor + N_FLOWS_LOOKAHEAD;
	if (lookaheadFlowIdx > g_totalFlows) lookaheadFlowIdx = g_totalFlows;
	uint32_t k = 0;
	while (flowToLoadCursor < lookaheadFlowIdx) {
		FlowPathToScheduleInfo& schItem = (*g_simStates->flowPathToScheduleInfo)[flowToLoadCursor];
		FlowLoader(schItem);
		++flowToLoadCursor;
		++g_totalFlowsLoaded;
		++k;
	}

	if (lookaheadFlowIdx < g_totalFlows) {
		double lastFlowAbsTime = (*g_simStates->flowPathToScheduleInfo)[flowToLoadCursor].event_time;
		Simulator::Schedule(Seconds(lastFlowAbsTime) - Simulator::Now(), &FlowsLoaderLazy);
	}

	cout << "==> LOADED " << k << " FLOWS @ " << Simulator::Now() << endl;
}

void SimulateFlowsByFlowPath() {
	g_totalFlowsLoaded = 0;
	g_flowFinished = 0;
	g_flowStarted = 0;
	g_totalFlows = g_simStates->flowPathToScheduleInfo->size();

	if (g_simStates->flowType == "tcp") {
		Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue(ns3::TcpNewReno::GetTypeId()));
		Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(true));

	} else if (g_simStates->flowType == "dctcp") {
		Config::SetDefault("ns3::TcpL4Protocol::SocketType",
				TypeIdValue(ns3::TcpDctcp::GetTypeId()));
		Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(true));
	}

	if (g_simStates->flowType == "tcp" || g_simStates->flowType == "dctcp") {
		Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (SEGMENT_SIZE));
		Config::SetDefault ("ns3::TcpSocket::DelAckCount", UintegerValue (2));
	}

	FlowsLoaderLazy();
}

void ProgressPrinter()
{
    double currentTime = Simulator::Now().GetSeconds();
    std::cout << "\rProgress: " << currentTime << "/" << g_totalSimulationTimeInSecs << " s completed, and " <<
    		g_flowStarted << "/" << g_flowFinished << "/" << g_totalFlowsLoaded << "/" << g_totalFlows <<
			" flows started, finished, loaded, and total" << std::flush;

    if (currentTime < g_totalSimulationTimeInSecs)
    {
        Simulator::Schedule(MilliSeconds(1.0), &ProgressPrinter);
    }
}

int main(int argc, char *argv[]) {
	string logPrefix = "/home/zby/dcqcn-net-control/test-topo/";
	string flowType = "dctcp";
	string red = "true";
	string portSrcTypeStr = "valid";  // ecmp or valid
	string pcapStr = "false"; // enable pcap
	string flowPathToScheduleInfoFileName = "flow-path-to-schedule-info.csv";
	string routesFileName = "routes";
	string outputFilePath = "/tmp/flow-perf.csv";

	CommandLine cmd;
	cmd.AddValue("logprefix", "log path prefix", logPrefix);
	cmd.AddValue("schedulefile", "flow-path-to-schedule-info file name", flowPathToScheduleInfoFileName);
	cmd.AddValue("routesfile", "routes file name", flowPathToScheduleInfoFileName);
	cmd.AddValue("portsrc", "port src type in ecmp or valid", portSrcTypeStr);
	cmd.AddValue("type", "tcp, dctcp or udp", flowType);
	cmd.AddValue("red", "true or false", red);
	cmd.AddValue("pcap", "true or false", pcapStr);
	cmd.AddValue("outfile", "output file", outputFilePath);
	cmd.Parse(argc, argv);

	if (flowType != "tcp" && flowType != "dctcp" && flowType != "udp") {
		NS_ABORT_MSG("Wrong parameter argument for --type");
	}
	if (red != "true" && red != "false") {
		NS_ABORT_MSG("Wrong parameter argument for --red");
	}
	if (pcapStr != "true" && pcapStr != "false") {
		NS_ABORT_MSG("Wrong parameter argument for --pcap");
	}

	string topologyFile = logPrefix + "topo";
	string portMapFile = logPrefix + "ports";
	string routesFile = logPrefix + routesFileName;
	string seedsFile = logPrefix + "seeds";
	string permutationsFile = logPrefix + "permutations";
	string flowPathToScheduleInfoFile = logPrefix + flowPathToScheduleInfoFileName;
	string linkRate = "10Gbps";
	string linkDelay = "2us";

	ParseTopology(topologyFile, nodeCount, epNodes, edges);
	ParsePortMap(portMapFile, portAssignments);
	CheckPortAssignmentsCoverage(edges, portAssignments);  // NEW VALIDATION STEP

	NodeContainer nodes;
	nodes.Create(nodeCount);

	//Ipv4StaticRoutingHelper staticRoutingHelper;
	//stack.SetRoutingHelper(staticRoutingHelper);

	MyIpv4EcmpRoutingHelper ecmpRoutingHelper;
	InternetStackHelper stack;
	stack.SetRoutingHelper(ecmpRoutingHelper);
	stack.Install(nodes);

	ConfigureSeedsAndPermutations(seedsFile, permutationsFile, nodes);

	PointToPointHelper p2p;
	p2p.SetDeviceAttribute("DataRate", StringValue(linkRate));
	p2p.SetChannelAttribute("Delay", StringValue(linkDelay));

	Ipv4AddressHelper ipv4;

	TrafficControlHelper tchRed10;
	if (red == "true") {
		Config::SetDefault ("ns3::RedQueueDisc::UseEcn", BooleanValue (true));
		// ARED may be used but the queueing delays will increase; it is disabled
		// here because the SIGCOMM paper did not mention it
		// Config::SetDefault ("ns3::RedQueueDisc::ARED", BooleanValue (true));
		// Config::SetDefault ("ns3::RedQueueDisc::Gentle", BooleanValue (true));
		Config::SetDefault ("ns3::RedQueueDisc::UseHardDrop", BooleanValue (false));
		Config::SetDefault ("ns3::RedQueueDisc::MeanPktSize", UintegerValue (1500));
		// Triumph and Scorpion switches used in DCTCP Paper have 4 MB of buffer
		// If every packet is 1500 bytes, 2666 packets can be stored in 4 MB
		Config::SetDefault ("ns3::RedQueueDisc::MaxSize", QueueSizeValue (QueueSize ("2666p")));
		// DCTCP tracks instantaneous queue length only; so set QW = 1
		Config::SetDefault ("ns3::RedQueueDisc::QW", DoubleValue (1));
		Config::SetDefault ("ns3::RedQueueDisc::MinTh", DoubleValue (20));
		Config::SetDefault ("ns3::RedQueueDisc::MaxTh", DoubleValue (60));

		// MinTh = 50, MaxTh = 150 recommended in ACM SIGCOMM 2010 DCTCP Paper
		// This yields a target (MinTh) queue depth of 60us at 10 Gb/s
		tchRed10.SetRootQueueDisc ("ns3::RedQueueDisc",
				"LinkBandwidth", StringValue (linkRate),
				"LinkDelay", StringValue (linkDelay),
				"MinTh", DoubleValue (50),
				"MaxTh", DoubleValue (150));
	}

	for (const auto &edge : edges) {
		NodeContainer nc = NodeContainer(nodes.Get(edge.src), nodes.Get(edge.dst));
		NetDeviceContainer ndc = p2p.Install(nc);
		if (flowType == "red") {
			tchRed10.Install (ndc);
		}

		Ptr<Ipv4> ipv4Src = nodes.Get(edge.src)->GetObject<Ipv4>();
		Ptr<Ipv4> ipv4Dst = nodes.Get(edge.dst)->GetObject<Ipv4>();

		Ptr<NetDevice> devSrc = ndc.Get(0);
		Ptr<NetDevice> devDst = ndc.Get(1);

		int32_t ifSrc = ipv4Src->GetInterfaceForDevice(devSrc);
		int32_t ifDst = ipv4Dst->GetInterfaceForDevice(devDst);

		if (ifSrc == -1) {
			ifSrc = ipv4Src->AddInterface(devSrc);
		}
		if (ifDst == -1) {
			ifDst = ipv4Dst->AddInterface(devDst);
		}

		devSrc->GetObject<PointToPointNetDevice>()->GetQueue()->TraceConnectWithoutContext(
				"MacTxDrop", MakeBoundCallback(&PacketDropPerDevice, edge.src, edge.dst));
		devSrc->GetObject<PointToPointNetDevice>()->GetQueue()->TraceConnectWithoutContext(
				"MacTx", MakeBoundCallback(&PacketSentPerDevice, edge.src, edge.dst));

		devDst->GetObject<PointToPointNetDevice>()->GetQueue()->TraceConnectWithoutContext(
				"MacTxDrop", MakeBoundCallback(&PacketDropPerDevice, edge.dst, edge.src));
		devDst->GetObject<PointToPointNetDevice>()->GetQueue()->TraceConnectWithoutContext(
				"MacTx", MakeBoundCallback(&PacketSentPerDevice, edge.dst, edge.src));

		ipv4Src->AddAddress(ifSrc, Ipv4InterfaceAddress(Ipv4Address(edge.ip1.c_str()), Ipv4Mask("255.255.255.0")));
		ipv4Dst->AddAddress(ifDst, Ipv4InterfaceAddress(Ipv4Address(edge.ip2.c_str()), Ipv4Mask("255.255.255.0")));

		ipv4Src->SetMetric(ifSrc, 1);
		ipv4Dst->SetMetric(ifDst, 1);

		ipv4Src->SetUp(ifSrc);
		ipv4Dst->SetUp(ifDst);

		portsMap[edge.src][edge.dst] = {portAssignments.at({edge.src, edge.dst}), devSrc};
		portsMap[edge.dst][edge.src] = {portAssignments.at({edge.dst, edge.src}), devDst};

		IntfInfo ii;
		ii.nodeID = edge.src;
		ii.portID = portAssignments.at({edge.src, edge.dst});
		ii.intfID = ifSrc;
		ipToIntfInfoMap[edge.ip1.c_str()] = ii;

		ii.nodeID = edge.dst;
		ii.portID = portAssignments.at({edge.dst, edge.src});
		ii.intfID = ifDst;
		ipToIntfInfoMap[edge.ip2.c_str()] = ii;
	}

	// Ipv4GlobalRoutingHelper::PopulateRoutingTables();

	ConfigureStaticRoutesFromFile(routesFile, nodes, portsMap);

	// Print a sample of the port map
	for (const auto& entry : portsMap) {
		const auto& src = entry.first;
		const auto& dstMap = entry.second;

		for (const auto& dstEntry : dstMap) {
			const auto& dst = dstEntry.first;
			const auto& port_dev = dstEntry.second;

			cout << "Node " << src << " -> Node " << dst
					<< " via port " << port_dev.first
					<< ", dev=" << port_dev.second << endl;
		}
	}

	DumpAllRoutingTables(nodes);

	map<uint32_t, Ipv4Address> nodeIpMap;
	for (uint32_t i = 0; i < nodes.GetN(); ++i)
	{
		Ptr<Ipv4> ipv4 = nodes.Get(i)->GetObject<Ipv4>();
		Ipv4Address ip = ipv4->GetAddress(1, 0).GetLocal(); // Assuming interface 1 is used
		nodeIpMap[i] = ip;
	}

	if (pcapStr == "true") {
		p2p.EnablePcapAll("dcn-pcap");
	}

	vector<FlowPathToScheduleInfo> flowPathToScheduleInfo = LoadFlowPathToScheduleFromFiles(flowPathToScheduleInfoFile);

	uint8_t portSrcType = portSrcTypeStr == "ecmp" ? PORT_SRC_ORIGNAL : PORT_SRC_VALID;

	g_simStates = new SimStates;
	g_simStates->flowPathToScheduleInfo = &flowPathToScheduleInfo;
	g_simStates->flowType = flowType;
	g_simStates->nodeIpMap = &nodeIpMap;
	g_simStates->nodes = &nodes;
	g_simStates->portSrcType = portSrcType;

	SimulateFlowsByFlowPath();


	Simulator::Schedule(MilliSeconds(1.0), &ProgressPrinter);

	Simulator::Stop(Seconds(g_totalSimulationTimeInSecs));
	Simulator::Run();
	Simulator::Destroy();

	// MyIpv4EcmpRouting::printTrackedPaths();

	WriteFlowPerfToCsv(outputFilePath);

	return 0;
}
