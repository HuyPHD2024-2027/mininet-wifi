#!/usr/bin/env python

'Setting position of the nodes'

import sys

from mininet.log import setLogLevel, info
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
import threading
import time

def monitor_contacts_and_position(node):
    """Monitor contact discovery and node positions"""
    while True:
        pos = node.position
        info(f"\n[{node.name}] Status Update:")
        info(f"\n  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        time.sleep(10)

def topology(args):

    net = Mininet_wifi()

    info("*** Creating nodes\n")
    ap1 = net.addAccessPoint('ap1', ssid='new-ssid', mode='g', channel='1',
                             failMode="standalone", mac='00:00:00:00:00:01',
                             position='50,50,0')
    sta1 = net.addStation('sta1', mac='00:00:00:00:00:02', ip='10.0.0.1/8',
                   position='30,60,0')
    sta2 = net.addStation('sta2', mac='00:00:00:00:00:03', ip='10.0.0.2/8',
                   position='70,30,0')
    h1 = net.addHost('h1', ip='10.0.0.3/8')

    info("*** Configuring propagation model\n")
    net.setPropagationModel(model="logDistance", exp=4.5)

    info("*** Configuring nodes\n")
    net.configureNodes()

    info("*** Creating links\n")
    net.addLink(ap1, h1)

    for node in [sta1, sta2]:
        thread = threading.Thread(target=monitor_contacts_and_position, args=(node,))
        thread.daemon = True
        thread.start()

    if '-p' not in args:
        net.plotGraph(max_x=100, max_y=100)

    info("*** Starting network\n")
    net.build()
    ap1.start([])

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    topology(sys.argv)
