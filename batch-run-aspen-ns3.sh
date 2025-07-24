#!/bin/bash

# Check if both parameters are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <k-ports-of-each-switch> <rack> <m-paths> <start> <end>"
    exit 1
fi

k_ports=$1
rack=$2
m_paths=$3
start_idx=$4
end_idx=$5
total=$((end_idx - start_idx + 1))

for ((i=start_idx; i<=end_idx; i++)); do
    dir="/home/zby/dcqcn-net-control/aspen-topo-k-${k_ports}-rack-${rack}-m-${m_paths}-fixed/aspen-topo-k-${k_ports}-rack-${rack}-m-${m_paths}-$i/"

    # Create directory if it doesn't exist
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi

    echo "[$((i-start_idx+1))/${total}] Running simulation for aspen-topo-k-${k_ports}-rack-${rack}-m-${m_paths}-$i..."

    python ./dcn_sim.py -t aspen_ns3 -k ${k_ports} -m ${m_paths} -n 1000 -c unicast -l "$dir" -r
done