#!/bin/bash

# Define parameter sets
k_values=(12 16)
rack_values=(20 40 60 80 100)
m_paths_values=(10 20 30 -1)
o_values=("valid" "ecmp")

# Base path
base_path="/home/zby/dcqcn-net-control"

# Calculate total number of combinations
total_steps=$(( ${#k_values[@]} * ${#rack_values[@]} * ${#m_paths_values[@]} * ${#o_values[@]} ))
current_step=1

# Loop over all combinations
for k in "${k_values[@]}"; do
    for rack in "${rack_values[@]}"; do
        for m_paths in "${m_paths_values[@]}"; do
          for o in "${o_values[@]}"; do
              echo "==================================================================="
              echo "Step $current_step / $total_steps: Running with k=${k}, rack=${rack}, o=${o}"
              topo_path="${base_path}/aspen-topo-k-${k}-rack-${rack}-m-${m_paths}-fixed/"
              echo "Command: python ./dcn_sim.py -t run_aspen_incast_batch_folders -l $topo_path -k $k -n 100 -o $o"
              echo "==================================================================="
              python ./dcn_sim.py -t run_aspen_incast_batch_folders -l "$topo_path" -k "$k" -a "${rack}" -m "${m_paths}" -n 100 -o "$o"
              echo ""
              ((current_step++))
          done
        done
    done
done

echo "All simulations completed successfully!"
