#!/usr/bin/python

from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.opportunisticNode import OpportunisticNode
from mn_wifi.cli import CLI
from mn_wifi.opportunisticLink import opportunisticLink
from mn_wifi.experiments import (
    run_delivery_experiment,
    run_convergence_experiment,
    run_resource_experiment
)
from mn_wifi.experiments import ExperimentMetrics
import time
import threading
import sys
import json
import matplotlib.pyplot as plt

def monitor_contacts_and_position(node):
    """Monitor contact discovery and node positions"""
    while True:
        pos = node.position
        info(f"\n[{node.name}] Status Update:")
        info(f"\n  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        info(f"\n  Bundles: {list(node.bundle_store.keys())}")
        info(f"\n  CRDT Counter: {node.rl_states_counter.value()}")
        info(f"\n  Packets Tracked: {len(node.forwarded_packets.get_elements())}")
        
        if node.contact_history:
            info("\n  Recent Contacts:")
            for contact in node.contact_history[-3:]:  # Show last 3 contacts
                info(f"\n    - {contact['node']} at RSSI: {contact['rssi']}dB")
                info(f"\n      Time: {time.strftime('%H:%M:%S', time.localtime(contact['timestamp']))}")
        info("\n" + "-"*40)
        time.sleep(30)

def demo_opportunistic_transfer(net):
    """Demonstrate opportunistic packet transfer between mobile nodes"""
    sta1, sta2, sta3 = net.get('sta1'), net.get('sta2'), net.get('sta3')
    
    # Start monitoring threads
    info("\n*** Starting node monitoring ***\n")
    for node in [sta1, sta2, sta3]:
        thread = threading.Thread(target=monitor_contacts_and_position, args=(node,))
        thread.daemon = True
        thread.start()

    info("\n*** Starting Opportunistic Network Demo ***\n")
    
    # Initial bundle creation
    info("\n[sta1] Creating initial data bundle 'message1'\n")
    sta1.store_bundle('message1', "Hello from sta1!", ttl=120)
    
    # Let nodes move and exchange data
    info("\n*** Nodes are moving according to RandomWayPoint model ***")
    info("\n*** Watch for node encounters and state updates ***\n")
    
    # Create periodic test messages
    message_counter = 0
    while True:
        try:
            if message_counter < 5:  # Limit to 5 test messages
                info(f"\n*** Creating test message {message_counter + 2} ***\n")
                sta1.store_bundle(f'message{message_counter + 2}', 
                                f"Test message {message_counter + 2}", 
                                ttl=120)
                message_counter += 1
            time.sleep(10)  # Create a new message every 10 seconds
        except KeyboardInterrupt:
            break

def run_all_experiments(net):
    """Run all experiments and generate plots"""
    info("\n*** Starting Comprehensive Experiments ***\n")
    
    # Run delivery vs density experiment
    info("\n*** Running Delivery Experiment ***\n")
    delivery_metrics = run_delivery_experiment(net)
    
    # Run CRDT convergence experiment
    info("\n*** Running Convergence Experiment ***\n")
    convergence_metrics = run_convergence_experiment(net)
    
    # Run resource usage experiment
    info("\n*** Running Resource Usage Experiment ***\n")
    resource_metrics = run_resource_experiment(net)
    
    # Combine metrics
    all_metrics = delivery_metrics
    all_metrics.convergence_times = convergence_metrics.convergence_times
    all_metrics.resource_usage.extend(resource_metrics.resource_usage)
    all_metrics.crdt_overhead.extend(resource_metrics.crdt_overhead)
    
    # Generate plots
    info("\n*** Generating Result Plots ***\n")
    all_metrics.plot_all_metrics()
    
    return all_metrics

def topology(args):
    "Create an opportunistic network topology with mobility"
    net = Mininet_wifi()
    
    info("*** Creating nodes\n")
    # Create nodes with initial positions spread across the area
    sta1 = net.addStation('sta1', cls=OpportunisticNode, position='10,10,0',
                         antennaHeight='1', antennaGain='5', max_packets=1)
    sta2 = net.addStation('sta2', cls=OpportunisticNode, position='50,50,0',
                         antennaHeight='1', antennaGain='5', max_packets=1)
    sta3 = net.addStation('sta3', cls=OpportunisticNode, position='90,90,0',
                         antennaHeight='1', antennaGain='5', max_packets=1)
        
    info("*** Configuring propagation model\n")
    net.setPropagationModel(model="logDistance", exp=4)
    
    info("*** Configuring Opportunistic nodes\n")
    net.configureWifiNodes()
    
    info("*** Creating links\n")
    # Create opportunistic links
    for sta in [sta1, sta2, sta3]:
        link = net.addLink(sta, intf=f'{sta.name}-wlan0', 
                          cls=opportunisticLink,
                          ssid='oppnet', 
                          mode='mesh', 
                          channel=5)
        # Configure opportunistic mode
        link.configure_opportunistic()
        
    if '-p' not in args:
        net.plotGraph(max_x=100, max_y=100)
    
    info("*** Configuring mobility model\n")
    net.setMobilityModel(time=0, model='RandomDirection',
                             max_x=100, max_y=100,
                             min_v=0.5, max_v=0.8, seed=20)
    
    info("*** Starting network\n")
    net.build()
    
    # Give time for neighbor discovery to initialize
    info("\n*** Waiting for neighbor discovery to initialize\n")
    time.sleep(5)

    # # Run experiments with visualization
    # run_all_experiments(net)
    
    info("\n*** Network is ready ***\n")
    CLI(net)
    
    info("*** Stopping network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology(sys.argv) 