import os
import numpy as np
import sys
import random as rand
import threading

cmd_template = ("./waf --run scratch/dcn-sim --command-template='%s --logprefix={log_prefix} "
                "--portsrc={port_type} --routesfile={routes_file} --type=dctcp --red=true --pcap=false --schedulefile={schedule_file_name} "
                "--outfile={output_file_path}'")
ns3_dir = '/home/zby/ext/ns-allinone-3.34/ns-3.34/'

class data_plane_ns3_sim:
    @staticmethod
    def run_batch_sim_in_parallel(task_func, n_threads, params_list):
        n_tasks = len(params_list)
        i = 0
        running_tasks = []
        for params in params_list:
            if len(running_tasks) == n_threads:
                for task_info in running_tasks:
                    task_info[1].join()
                    print("===================> TASK FINISHED " + str(task_info[0]) + "/" + str(n_tasks) +
                          " with the paramters of " + str(task_info[2]))
                running_tasks = []

            t = threading.Thread(target=task_func, args=params)
            t.setDaemon(True)
            t.start()
            running_tasks.append([i, t, params])
            i = i + 1

        for task_info in running_tasks:
            task_info[1].join()
            print("===================> TASK FINISHED " + str(task_info[0]) + "/" + str(n_tasks) +
                  " with the paramters of " + str(task_info[2]))

    @staticmethod
    def run_dcn_sim(log_prefix, port_type, schedule_file_name, output_file_path, routes_file_name, renew):
        if not renew:
            if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
                print("==> PROGRESS automatic jump the old files")
                return

        cmd = cmd_template.format(log_prefix=log_prefix, port_type=port_type, routes_file=routes_file_name,
                                  schedule_file_name=schedule_file_name, output_file_path=output_file_path)
        os.system("xterm -e \"cd " + ns3_dir + " && " + cmd + "\"")
        print("==> PROGRESS: the last configuration with schedule_file_name (" + schedule_file_name + ") is finished.")

    @staticmethod
    def run_batch_dcn_sim_incast(log_prefix, port_type, n_times, n_threads, renew):
        num_racks_all = [x for x in range(30, 40, 10)]
        template_flow_path_to_schedule_file_name = "flow-path-to-schedule-info-rack-{num_racks}-times-{times}.csv"
        template_output_file_path = log_prefix + "flow-perf-port-{port_type}-rack-{num_racks}-times-{times}.csv"
        #template_routes_file_name = "routes-rack-{num_racks}-times-{times}.csv"

        params_list = []
        for num_racks in num_racks_all:
            for times in range(n_times):
                schedule_file_name = template_flow_path_to_schedule_file_name.format(num_racks=num_racks,
                                                                                     times=times)
                output_file_path = template_output_file_path.format(port_type=port_type, num_racks=num_racks, times=times)
                routes_file_name = "routes"

                params = (log_prefix, port_type, schedule_file_name, output_file_path, routes_file_name, renew)
                params_list.append(params)

        data_plane_ns3_sim.run_batch_sim_in_parallel(data_plane_ns3_sim.run_dcn_sim, n_threads, params_list)

    @staticmethod
    def run_batch_dcn_sim_incast_for_folders(log_prefix, port_type, num_rack, times_start, times_end,
                                             n_threads, renew,
                                             subfolder_template="test-topo-rack-{num_rack}-{i}", m_paths=None):
        flow_path_to_schedule_file_name = "flow-path-to-schedule-info.csv"
        template_output_file_name = "flow-perf-port-{port_type}.csv"
        template_folder_path = log_prefix + subfolder_template + "/"
        routes_file_name = "routes"

        params_list = []
        for i in range(times_start, times_end + 1):
            if m_paths is not None:
                folder_path = template_folder_path.format(num_rack=num_rack, m_paths=m_paths, i=i)
            else:
                folder_path = template_folder_path.format(num_rack=num_rack, i=i)
            output_file_path = folder_path + template_output_file_name.format(port_type=port_type)
            params = (folder_path, port_type, flow_path_to_schedule_file_name, output_file_path, routes_file_name, renew)
            params_list.append(params)

        data_plane_ns3_sim.run_batch_sim_in_parallel(data_plane_ns3_sim.run_dcn_sim, n_threads, params_list)