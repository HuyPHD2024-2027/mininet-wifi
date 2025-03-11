#!/usr/bin/python

from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.opportunisticNode import OpportunisticNode
from mn_wifi.cli import CLI
from mn_wifi.opportunisticLink import opportunisticLink
from mn_wifi.experiments import (
    run_comprehensive_benchmark
)
import time
import sys

def topology(args):
    "Create an opportunistic network topology with mobility"
    net = Mininet_wifi()
    
    info("*** Creating nodes\n")
    # Create nodes with initial positions spread across the area
    sta1 = net.addStation('sta1', cls=OpportunisticNode, position='10,10,0',
                         antennaHeight='1', antennaGain='5', max_packets=5, max_crdt_packets=5)
    sta2 = net.addStation('sta2', cls=OpportunisticNode, position='50,50,0',
                         antennaHeight='1', antennaGain='5', max_packets=5, max_crdt_packets=5)
    sta3 = net.addStation('sta3', cls=OpportunisticNode, position='90,90,0',
                         antennaHeight='1', antennaGain='5', max_packets=5, max_crdt_packets=5)
        
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
        crdt = True 
        link.configure_opportunistic(crdt)
        
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
    # run_comprehensive_benchmark(net, duration=300)  # Run for 5 minutes
    
    info("\n*** Network is ready ***\n")
    CLI(net)
    
    info("*** Stopping network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology(sys.argv) 