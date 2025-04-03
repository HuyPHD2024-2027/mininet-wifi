import time
from mininet.log import info, error
from mn_wifi.node import Node_wifi
from mn_wifi.forwarding_agent import ForwardingAgent
from mn_wifi.link import adhoc
from mn_wifi.constants import STATION_MAC_MAPPING, get_node_by_mac, register_station_mac
from mn_wifi.opportunisticNode import OpportunisticNode
import threading
import math
import socket
import json
import struct

class opportunisticLink(adhoc):
    """Link for opportunistic networking between nodes"""
    
    def __init__(self, node, intf=None, **params):
        super(opportunisticLink, self).__init__(node, intf, **params)
        self.node = node
        self.intf = intf
        self.name = params.get('intf', f'{node.name}-wlan0')
        self.beacon_interval = params.get('beacon_interval', 1)  # Reduced from 5 to 1 second for more responsiveness
        self.last_encounters = {}
        self.discovery_thread = None
        self.broadcast_thread = None
        self.queue_length = params.get('queue_length', 10)
        self.hop_count = params.get('hop_count', 5)
        self.ttl = params.get('ttl', 60)
        self.range_threshold = params.get('range_threshold', 50)  # Default range threshold in meters
        self.broadcast_port = params.get('broadcast_port', 12345)  # Port for broadcast messages
        self.discovered_neighbors = set()  # Set to track discovered neighbors
        # Register node's MAC
        if intf and hasattr(intf, 'mac'):
            register_station_mac(node.name, intf.mac)
        
    
    def discover_neighbors(self, crdt=False):
        """Discover neighboring nodes using broadcast mechanism"""
        # Wait a few seconds to ensure network is ready
        time.sleep(5)
        
        # Start broadcast thread
        self.start_broadcast()
        
        # Start listening for broadcasts
        self.listen_for_broadcasts(crdt)
    
    def start_broadcast(self):
        """Start broadcasting node presence"""
        if not self.broadcast_thread:
            self.broadcast_thread = threading.Thread(target=self._broadcast_presence)
            self.broadcast_thread.daemon = True
            self.broadcast_thread.start()
            info(f"*** Started broadcast for {self.node.name}\n")
    
    def _broadcast_presence(self):
        """Broadcast node presence periodically"""
        try:
            # Create UDP socket for broadcasting
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            # Set socket timeout
            sock.settimeout(0.1)
            
            while True:
                try:
                    # Check if node is still active
                    if not hasattr(self.node, 'shell') or not self.node.shell:
                        info(f"Node {self.node.name} is no longer active, stopping broadcast\n")
                        break
                    
                    # Create broadcast message
                    message = {
                        'type': 'beacon',
                        'node_name': self.node.name,
                        'mac': self.node.wintfs[0].mac if hasattr(self.node, 'wintfs') and self.node.wintfs else None,
                        'position': self.node.position if hasattr(self.node, 'position') else None,
                        'timestamp': time.time()
                    }
                    
                    # Convert to JSON and encode
                    data = json.dumps(message).encode('utf-8')
                    
                    # Broadcast message
                    sock.sendto(data, ('<broadcast>', self.broadcast_port))
                    
                    # Sleep for beacon interval
                    time.sleep(self.beacon_interval)
                except Exception as e:
                    error(f"Error in broadcast: {str(e)}\n")
                    time.sleep(1)
        except Exception as e:
            error(f"Error creating broadcast socket: {str(e)}\n")
    
    def listen_for_broadcasts(self, crdt=False):
        """Listen for broadcast messages from other nodes"""
        try:
            # Create UDP socket for listening
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', self.broadcast_port))
            
            # Set socket timeout
            sock.settimeout(0.1)
            
            while True:
                try:
                    # Check if node is still active
                    if not hasattr(self.node, 'shell') or not self.node.shell:
                        info(f"Node {self.node.name} is no longer active, stopping listener\n")
                        break
                    
                    # Receive data
                    data, addr = sock.recvfrom(65535)
                    
                    # Decode and parse message
                    message = json.loads(data.decode('utf-8'))
                    
                    # Process beacon message
                    if message.get('type') == 'beacon':
                        self._process_beacon(message, crdt)
                except socket.timeout:
                    # Timeout is expected, continue listening
                    continue
                except Exception as e:
                    error(f"Error in broadcast listener: {str(e)}\n")
                    time.sleep(1)
        except Exception as e:
            error(f"Error creating listener socket: {str(e)}\n")
    
    def _process_beacon(self, message, crdt=False):
        """Process a received beacon message"""
        try:
            # Extract message data
            node_name = message.get('node_name')
            mac = message.get('mac')
            position = message.get('position')
            timestamp = message.get('timestamp', 0)
            
            # Skip if it's our own beacon
            if node_name == self.node.name:
                return
            
            # Find the neighbor node
            neighbor = None
            
            # Try to find by name first
            if hasattr(self.node, 'net') and self.node.net:
                for node in self.node.net.stations:
                    if node.name == node_name:
                        neighbor = node
                        break
            
            # If not found by name, try to find by MAC
            if not neighbor and mac:
                neighbor = self._find_node_by_mac(mac)
            
            # Check if we've recently processed this encounter
            current_time = time.time()
            last_time = self.last_encounters.get(neighbor.name, 0)
            
            if current_time - last_time > self.beacon_interval:
                # Calculate RSSI based on position if available
                rssi = -100  # Default RSSI
                
                if position and hasattr(self.node, 'position') and self.node.position:
                    distance = self._calculate_distance(self.node.position, position)
                    rssi = self._calculate_rssi(distance)
                
                info(f"*** {self.node.name} discovered {neighbor.name} via broadcast (RSSI: {rssi}dB)\n")
                # self.handle_encounter(neighbor, rssi, crdt)
                # self.last_encounters[neighbor.name] = current_time
        except Exception as e:
            error(f"Error processing beacon: {str(e)}\n")

    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        try:
            x1, y1, z1 = float(pos1[0]), float(pos1[1]), float(pos1[2])
            x2, y2, z2 = float(pos2[0]), float(pos2[1]), float(pos2[2])
            return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        except Exception as e:
            error(f"Error calculating distance: {str(e)}\n")
            return float('inf')  # Return infinity if calculation fails
    
    def _calculate_rssi(self, distance):
        """Calculate RSSI based on distance using a simplified path loss model"""
        # Simple path loss model: RSSI = -10 * n * log10(d) + C
        # where n is the path loss exponent (typically 2-4) and C is a constant
        n = 3.5  # Path loss exponent
        C = -30  # Reference signal strength at 1m
        if distance < 0.1:  # Avoid log(0)
            distance = 0.1
        return -10 * n * math.log10(distance) + C
    
    def handle_encounter(self, neighbor, rssi, crdt=False):
        """Handle encountering another node"""
        try:
            # Update contact history
            self.node.update_network_state(neighbor, rssi)
            
            # 1. Exchange CRDT data
            if crdt:
                self.node.send_crdt_update(neighbor.name)
                
            # 2. Check bundles that might be deliverable
            self.node._check_bundles(neighbor, crdt)

            info(f"*** {self.node.name} completed encounter with {neighbor.name}\n")
        
        except Exception as e:
            error(f"Error in handle_encounter: {str(e)}\n")
            import traceback
            error(traceback.format_exc())

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
