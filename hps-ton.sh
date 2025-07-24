#!/bin/bash

results_dir=/home/zby/dcqcn-net-control/results-ton/
if [ ! -d ${results_dir} ]; then
  mkdir ${results_dir}
  #rm -rf ${results_dir}/*
fi
n_pairs=1000
python=python3
$python ./dcn_sim.py -t parallel_fattree -n${n_pairs} -c unicast -l ${results_dir} -r
$python ./dcn_sim.py -t batch_size_fattree -n${n_pairs} -c unicast -l ${results_dir} -r
