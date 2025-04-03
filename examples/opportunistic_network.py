#!/usr/bin/python

from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.opportunisticNode import OpportunisticNode
from mn_wifi.cli import CLI
from mn_wifi.opportunisticLink import opportunisticLink
from mn_wifi.utils.saver import OpportunisticSaver
import time
import sys
import argparse
import os

def topology(args):
    "Create an opportunistic network topology with mobility"
    net = Mininet_wifi()
    
    info("*** Creating nodes\n")
    # Create nodes with initial positions spread across the area
    nodes = []
    for i in range(args.nodes):
        # Calculate position to spread nodes across the area
        x = (i * 90 / (args.nodes - 1)) + 10 if args.nodes > 1 else 50
        y = (i * 90 / (args.nodes - 1)) + 10 if args.nodes > 1 else 50
        
        node = net.addStation(f'sta{i+1}', 
                            cls=OpportunisticNode, 
                            min_x=0, max_x=1000, min_y=0, max_y=1000,
                            antennaHeight='1', 
                            antennaGain='5',
                            max_packets=args.max_packets if args.max_packets > 0 else float('inf'),
                            max_crdt_packets=args.max_crdt_packets if args.max_crdt_packets > 0 else float('inf'))
        nodes.append(node)
    
    # Initialize saver
    experiment_name = f"experiment_{args.nodes}nodes_{args.experiment_time}s"
    saver = OpportunisticSaver(experiment_name=experiment_name)
    
    # Register saver with each node for data collection
    for node in nodes:
        node.saver = saver
        
    info("*** Configuring propagation model\n")
    net.setPropagationModel(model="logDistance", exp=4)
    
    info("*** Configuring Opportunistic nodes\n")
    net.configureWifiNodes()
    
    info("*** Creating links\n")
    # Create opportunistic links
    for node in nodes:
        link = net.addLink(node, intf=f'{node.name}-wlan0', 
                          cls=opportunisticLink,
                          ssid='oppnet', 
                          mode='mesh', 
                          channel=5,
                          range_threshold=args.range_threshold,  # Add range threshold parameter
                          beacon_interval=args.beacon_interval)  # Add beacon interval parameter
        # Configure opportunistic mode
        link.configure_opportunistic(crdt=True)
        
    if not args.no_plot:
        net.plotGraph(max_x=1000, max_y=1000)
    
    info("*** Configuring mobility model\n")
    # Set the mobility model with explicit parameters
    net.setMobilityModel(time=0, model='RandomDirection',
                        max_x=1000, max_y=1000,
                        min_v=0.5, max_v=0.8, seed=20)
    
    info("*** Starting network\n")
    net.build()
    
    # Give time for neighbor discovery to initialize
    info("\n*** Waiting for neighbor discovery to initialize\n")
    time.sleep(5)

    if args.experiment_time > 0:
        # Create experiment configuration
        experiment_config = {
            'num_nodes': args.nodes,
            'duration': args.experiment_time,
            'max_packets': args.max_packets,
            'max_crdt_packets': args.max_crdt_packets,
            'mobility_model': 'RandomDirection',
            'mobility_params': {
                'max_x': 1000,
                'max_y': 1000,
                'min_v': 0.5,
                'max_v': 0.8,
                'seed': 20
            },
            'range_threshold': args.range_threshold,
            'beacon_interval': args.beacon_interval
        }
        
        info(f"\n*** Running experiment for {args.experiment_time} seconds ***\n")
        
        # Run the experiment
        time.sleep(args.experiment_time)
        
        # Generate graphs and save results
        saver.generate_performance_graphs()
        results_dir = saver.save_experiment_results(experiment_config)
        
        info(f"\n*** Experiment completed. Results saved to: {results_dir} ***\n")
    else:
        info("\n*** Network is ready ***\n")
        CLI(net)
    
    info("*** Stopping network\n")
    net.stop()

def parse_args():
    parser = argparse.ArgumentParser(description='Opportunistic Network Simulation')
    parser.add_argument('--nodes', type=int, default=3,
                      help='Number of nodes in the network (default: 3)')
    parser.add_argument('--experiment-time', type=int, default=0,
                      help='Duration of experiment in seconds (0 for CLI mode, default: 0)')
    parser.add_argument('--max-packets', type=int, default=0,
                      help='Maximum number of packets per node (0 for unlimited, default: 0)')
    parser.add_argument('--max-crdt-packets', type=int, default=0,
                      help='Maximum number of CRDT packets per node (0 for unlimited, default: 0)')
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable graph plotting')
    parser.add_argument('--range-threshold', type=float, default=50.0,
                      help='Range threshold in meters for neighbor discovery (default: 50.0)')
    parser.add_argument('--beacon-interval', type=float, default=1.0,
                      help='Beacon interval in seconds for neighbor discovery (default: 1.0)')
    return parser.parse_args()

if __name__ == '__main__':
    setLogLevel('info')
    args = parse_args()
    topology(args) 