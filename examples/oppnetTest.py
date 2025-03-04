#!/usr/bin/python
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.opportunisticNode import OpportunisticNode
from mn_wifi.opportunisticLink import opportunisticLink
from mn_wifi.cli import CLI
import time

def topology():
    "Create a network with opportunistic nodes"
    net = Mininet_wifi()
    
    info("*** Creating nodes\n")
    sta1 = net.addStation('sta1', cls=OpportunisticNode, position='10,10,0')
    sta2 = net.addStation('sta2', cls=OpportunisticNode, position='50,50,0')
    sta3 = net.addStation('sta3', cls=OpportunisticNode, position='100,100,0')
    
    info("*** Configuring propagation model\n")
    net.setPropagationModel(model="logDistance", exp=4)
    
    info("*** Configuring Opportunistic nodes\n")
    net.configureWifiNodes()
    
    info("*** Creating links\n")
    for sta in [sta1, sta2, sta3]:
        link = net.addLink(sta, intf=f'{sta.name}-wlan0', 
                           cls=opportunisticLink,
                           ssid='oppnet', 
                           mode='mesh', 
                           channel=5)
    
    info("*** Starting network\n")
    net.build()
    
    # Start the network and create some test bundles
    sta1.store_bundle('test1', "Hello from sta1!", ttl=120)
    
    # Let nodes move and exchange data
    info("*** Running mobility model\n")
    net.startMobility(startTime=0, model='RandomWayPoint', 
                     max_x=100, max_y=100, min_v=0.5, max_v=0.8)
    
    # Start CLI
    CLI(net)
    
    # Stop network
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology()