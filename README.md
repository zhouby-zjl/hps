# Source Code for Host-Based Path Selection and Scalable Power-of-Two Routing in Data Center Networks: HPS and PowerPath

## What is HPS?
In data center networks (DCNs), path selection via equal-cost multi-path (ECMP) hashing often leads to path overlap and port overuse, degrading performance. While recent techniques let hosts influence egress port selection through single-bit header changes, they offer limited coverage and are restricted to one-hop control. We propose a host-based path selector (HPS) that enables source hosts to steer packets across multiple hops by applying targeted multi-bit header modifications. HPS relies on two key principles: (a) leveraging predictable relationships in relative hash changes to control egress ports, and (b) applying criteria that ensure header changes guide packets along the intended path. HPS adjusts headers only at selected hops with diverging next-hop options, using test packets to compute valid modifications. This yields efficient, accurate path selection with polynomial-time complexity, constant space, and optional parallel acceleration for large-scale scalability. We evaluated HPS on two- to five-layer DCN topologies with up to 1000 random paths and varied ECMP hash functions. HPS consistently achieved high selection accuracy and low runtime, reducing path overlap by up to 36.6\% over ECMP and outperforming RePaC. NS-3 simulations further validated its ability to mitigate DCTCP incast congestion and enhance transport performance.

### Environment Setup for HPS Controller with NS-3 Data Plane
The HPS controller is implemented in Python and is capable of generating valid, randomized source port values for flow headers to steer packets along specific paths. To evaluate the effectiveness of HPS, it is integrated with the NS-3 network simulator, which emulates a data center network (DCN) environment using DCTCP-based traffic flows. To set up this environment, follow these steps:

(a) Run pip install -r requirements.txt to install the required Python dependencies.
(b) Download NS-3 from https://www.nsnam.org/ (version 3.34 is verified for compatibility). Then copy dcn-sim.cc into the ns3-3.XX/scratch/ directory.
(c) Edit data_plane_ns3_sim.py to update the ns3_dir variable to the correct NS-3 directory path. Modify exe-batch-aspen-ns3-sim.sh to set base_path to the HPS controller path. Also, update batch-run-aspen-ns3.sh to set the dir variable to the output log directory of the HPS controller.

### How to Run HPS?
**Single Case Test:**
1. Run the following command to perform a single-case test: python ./dcn_sim.py -t aspen -k ${k_ports} -b ${batch_size} -p ${n_cpus} -n ${n_pairs}  -c unicast -l "${dir}"
Here, k_ports specifies the number of ports per switch in the network; batch_size defines the number of XORed criteria combinations evaluated in each parallel execution to identify a valid source port modification; n_cpus indicates the number of CPU cores allocated for the computation; n_pairs determines the number of randomly selected communication pairs used in the test; and ${dir} denotes the directory where the output logs will be stored. Additionally, -t aspen sets the network topology to Aspen, though other options such as spine-leaf or fattree are also supported. For a full list of parameters and usage details, you can run:
python ./dcn_sim.py --help
2. Check the logs under ${dir}, which contain detailed information on network topology, flow characteristics, and path planning statistics.

**Batch Tests with NS-3**
1. To conduct comprehensive testing of the HPS controller, execute exe-batch-aspen-ns3-sim.sh. You may customize the test parameters directly within the script. This process will iterate over various network configurations and generate a series of corresponding logs.
2. Once the logs are generated, run the NS-3 data plane using the batch-run-aspen-ns3.sh script. Make sure that the specified log directory (dir) matches exactly with the one used during HPS controller execution.
3. The resulting NS-3 logs will be placed in the same directories as the HPS controller logs, providing detailed insights into transport-layer performance under the tested scenarios.

### Papers
[1] Boyang Zhou, Chunming Wu and Qiang Yang, "Enabling Source Hosts to Precisely Select Paths via ECMP Hash Linearity in Data Center Networks," 2024 IEEE International Conference on High Performance Computing and Communications (HPCC), Wuhan, China, 2024, pp. 1633-1642, doi: 10.1109/HPCC64274.2024.00216. Available at: https://ieeexplore.ieee.org/document/11083286 

[2] Boyang Zhou, Chunming Wu, etc., "Precise and Scalable Host-Based Path Selection via ECMP Hash Linearity in Data Center Networks," IEEE/ACM Transactions on Networking, 2025 (an extended version of HPCC paper, currently under submission)


## What is PowerPath?
In modern multi-layer data center networks, a key challenge for ECMP-based routing is ensuring that each in-path hop offers a power-of-two number of next hops—a requirement for preserving ECMP hash linearity and enabling deterministic host-side path selection. This constraint is frequently violated in irregular or dynamic topologies, calling for constraint-aware path generation. However, the problem is non-trivial: it is NP-complete via reduction from SAT. To address this, we propose PowerPath, a scalable and deployable path generation scheme that enforces the power-of-two constraint while promoting balanced link utilization. Rooted at the source, PowerPath performs lightweight breadth-first traversals to expose path connectivity, followed by randomized allocation and controlled branching to construct constraint-compliant paths—all within polynomial time and with low routing overhead. We evaluate PowerPath in NS-3 on large-scale Aspen topologies with 8–16-port switches, demonstrating strict constraint adherence, scalable routing table sizes, and substantial transport-layer improvements. When combined with host-side disjoint path selection, PowerPath achieves up to 8.05\% mean and 24.61\% tail goodput gains over traditional ECMP.

### Where is PowerPath and How is it Integrated in HPS controller?
PowerPath is implemented in the path_generator_with_ecmp_hash_linearity.py module. It is invoked by the generate_and_install_paths_ensuring_ecmp_linearity function in dcn_networks.py, which is further utilized in dcn_sim.py to perform simulation-based evaluations.

### How to Run PowerPath?
Before running PowerPath, ensure that the HPS controller is properly set up as described in the instructions above. Then proceed with the following steps:
(a) Open dcn_sim.py and set the parameter use_routing_linearity=True to enable PowerPath to generate paths that comply with ECMP hash linearity constraints.
(b) Follow the batch testing procedure described earlier to evaluate PowerPath's performance within the NS-3 simulation environment.

### Paper
[1] Boyang Zhou, Chunming Wu, etc., "Scalable Power-of-Two Routing for Host-Driven ECMP Path Selection in Data Center Networks," IEEE/ACM Transactions on Networking, 2025 (currently under submission)


 *********************************************************************************
This work is licensed under CC BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Copyright (c) 2025 Boyang Zhou

This file is a part of "Host-Based Path Selector via ECMP Hash Linearity in Data Center Networks"
(https://github.com/zhouby-zjl/hps/).

 **********************************************************************************
