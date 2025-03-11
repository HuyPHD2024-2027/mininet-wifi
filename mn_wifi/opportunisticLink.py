import time
from mininet.log import info, error
from mn_wifi.node import Node_wifi
from mn_wifi.forwarding_agent import ForwardingAgent
from mn_wifi.link import adhoc
from mn_wifi.constants import STATION_MAC_MAPPING, get_node_by_mac, register_station_mac
from mn_wifi.opportunisticNode import OpportunisticNode
import threading

class opportunisticLink(adhoc):
    """Link for opportunistic networking between nodes"""
    
    def __init__(self, node, intf=None, **params):
        super(opportunisticLink, self).__init__(node, intf, **params)
        self.node = node
        self.intf = intf
        self.name = params.get('intf', f'{node.name}-wlan0')
        self.discovery_interval = params.get('discovery_interval', 5)
        self.last_encounters = {}
        self.discovery_thread = None

        # Register node's MAC
        if intf and hasattr(intf, 'mac'):
            register_station_mac(node.name, intf.mac)
        
    
    def discover_neighbors(self, crdt=False):
        """Discover neighboring nodes"""
        # Wait a few seconds to ensure network is ready
        time.sleep(5)
        
        while True:
            try:
                # Check if node is still active
                if not hasattr(self.node, 'shell') or not self.node.shell:
                    info(f"Node {self.node.name} is no longer active, stopping discovery\n")
                    break
                
                # Get mesh peers with proper interface name
                mesh_info = self.node.cmd(f'iw dev {self.name} station dump')
                current_time = time.time()
                
                # Extract MACs and signal strengths
                lines = mesh_info.split('\n')
                for i, line in enumerate(lines):
                    if 'Station' in line:
                        try:
                            mac = line.split('Station')[1].strip().split()[0]
                            
                            # Get RSSI
                            rssi = -100
                            if i + 2 < len(lines):
                                signal_line = lines[i + 2]
                                if 'signal:' in signal_line:
                                    rssi = int(signal_line.split()[1])
                                    
                            # Find neighbor node
                            neighbor = get_node_by_mac(mac)
                            if neighbor:
                                last_time = self.last_encounters.get(neighbor.name, 0)
                                if current_time - last_time > self.discovery_interval:
                                    info(f"*** {self.node.name} discovered {neighbor.name} (RSSI: {rssi}dB)\n")
                                    self.node.handle_encounter(neighbor, rssi, crdt)
                                    self.last_encounters[neighbor.name] = current_time
                        except Exception as e:
                            error(f"Error processing station: {str(e)}\n")
                
                time.sleep(self.discovery_interval)
            except Exception as e:
                error(f"Error in discovery: {str(e)}\n")
                time.sleep(1)
    

    def configure_opportunistic(self, crdt=False):
        """Configure interface for opportunistic networking"""
        # Configure mesh interface
        intf = self.node.params['wlan'][0]
        
        # Use cmd instead of sendCmd to wait for completion
        self.node.cmd('ip link set {} down'.format(self.name))
        self.node.cmd('iw dev {} set type mesh'.format(self.name))
        self.node.cmd('ip link set {} up'.format(self.name))
        self.node.cmd('iw dev {} mesh join oppnet'.format(self.name))
        self.node.cmd('iw dev {} set channel {}'.format(self.name, self.channel))
        
        # Set mesh parameters
        self.node.cmd('iw dev {} set mesh_param mesh_hwmp_rootmode=0'.format(self.name))
        self.node.cmd('iw dev {} set mesh_param mesh_path_refresh_time=1000'.format(self.name))
        
        # Give some time for the interface to be ready
        time.sleep(1)
        
        # Start neighbor discovery thread
        self.start_discovery(crdt)
        
    def start_discovery(self, crdt=False):
        """Start the neighbor discovery thread"""
        if not self.discovery_thread:
            import threading
            self.discovery_thread = threading.Thread(target=self.discover_neighbors, args=(crdt,))
            self.discovery_thread.daemon = True
            self.discovery_thread.start()
            if crdt:
                info(f"*** Started neighbor discovery for {self.node.name} with CRDTs\n")
            else:
                info(f"*** Started neighbor discovery for {self.node.name}\n")
    def _find_node_by_mac(self, mac):
        """Find node object by MAC address using the global registry"""
        try:
            node = get_node_by_mac(mac)
            if node:
                info(f"{self.node.name} Found station {node.name} for MAC {mac}\n")
                return node
            # else:
            #     info(f"\nNo node found for MAC {mac}\n")
            #     return None
        except Exception as e:
            error(f"Error finding node for MAC {mac}: {str(e)}\n")
            import traceback
            error(traceback.format_exc())
            return None
