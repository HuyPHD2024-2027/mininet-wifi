#!/usr/bin/python

from mn_wifi.node import Station
from mn_wifi.crdt import GCounter, OrSet, NodeInfo
from mn_wifi.packet import Packet, PacketType
from mn_wifi.constants import NODE_REGISTRY, register_node, get_node_by_name
from mn_wifi.utils.network import PortManager, send_packet
from mn_wifi.utils.routing import is_good_next_hop
from mininet.log import info, error
import time
import json
import random
from threading import Thread, Lock
import socket
from socket import SO_REUSEADDR, SOL_SOCKET
from queue import Queue
from mn_wifi.utils.logging import logger

class OpportunisticNode(Station):
    """Station with opportunistic networking capabilities"""
    
    # Class variable for port management
    _used_ports = set()
    _port_lock = Lock()
    
    def __init__(self, name, **params):
        super(OpportunisticNode, self).__init__(name, **params)
        
        # Register this node
        register_node(name, self)
        
        # CRDT data structures
        self.bundle_store = Queue()
        self.network_state = GCounter()
        self.forwarded_packets = OrSet()

        # Network settings
        self.port = params.get('port') or PortManager.get_available_port()
        # Set to float('inf') for unlimited packets
        self.max_packets = params.get('max_packets', float('inf'))
        self.max_crdt_packets = params.get('max_crdt_packets', float('inf'))
        
        self.packets_sent = 0
        self.crdt_packets_sent = 0
        self.sequence_counter = 0
        
        # Start packet listener
        self.listening = True
        self.listener_thread = Thread(target=self._packet_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
        logger.log_event("Node Initialization", {
            "Name": self.name,
            "Port": self.port,
            "Max Packets": "Unlimited" if self.max_packets == float('inf') else self.max_packets,
            "Max CRDT Packets": "Unlimited" if self.max_crdt_packets == float('inf') else self.max_crdt_packets
        })
    
    def _packet_listener(self):
        """Listen for incoming packets"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(('', self.port))
            sock.settimeout(1)
            
            while self.listening:
                try:
                    data, addr = sock.recvfrom(65535)
                    packet_dict = json.loads(data.decode())
                    self._process_packet(packet_dict)
                except socket.timeout:
                    continue
                except Exception as e:
                    error(f"Listener error: {str(e)}\n")
        finally:
            sock.close()
    
    def _process_packet(self, packet):
        """Process received packet based on type"""
        if packet['packet_type'] == "CRDT":
            self._handle_crdt_packet(packet)
        elif packet['packet_type'] == "DATA":
            self._handle_data_packet(packet)
    
    def _handle_crdt_packet(self, packet):
        """Handle received CRDT packet"""
        try:
            if not packet['payload'] or not isinstance(packet['payload'], dict):
                logger.log_event("CRDT Error", {
                    "Source": packet['source'],
                    "Error": "Invalid payload"
                }, is_error=True)
                return
            
            crdt_data = packet['payload']
            
            # Merge counter
            if 'network_state' in crdt_data:
                other_counter = GCounter.from_dict(crdt_data['network_state'])
                self.network_state.merge(other_counter)
            
            # Merge received packets
            if 'forwarded_packets' in crdt_data:
                other_packets = OrSet()
                other_packets.elements = crdt_data['forwarded_packets']
                self.forwarded_packets.merge(other_packets)
            
            merged = {
                'network_state': self.network_state.to_dict(),
                'forwarded_packets': self.forwarded_packets.elements
            }
            
            logger.log_event("CRDT Sync", {
                "Source": packet['source'],
                "Destination": self.name,
                "Status": "Success",
                # "Merged": merged
            })
                
        except Exception as e:
            logger.log_event("CRDT Error", {
                "Source": packet['source'],
                "Error": str(e)
            }, is_error=True)
    
    def _handle_data_packet(self, packet):
        """Handle data packet"""
        if packet['destination'] == self.name and packet['destination'] != 'broadcast':
            delay = time.time() - packet['timestamp']
            if hasattr(self, 'saver'):
                self.saver.log_packet_delivery(
                    packet['source'], 
                    packet['destination'],
                    delay,
                    True
                )
            logger.log_event("Packet Delivery", {
                "Packet ID": packet['packet_id'],
                "From": packet['source'],
                "To": packet['destination'],
                "Delay": f"{delay:.2f}s"
            })
        else:
            # Store data in queue
            if isinstance(packet['payload'], dict):
                self.bundle_store.put({
                    'packet': packet,
                    'stored_at': time.time(),
                    'ttl': 300  # 5 minutes TTL default
                })
                logger.log_event("Packet Storage", {
                    "Packet ID": packet['packet_id'],
                    "From": packet['source'],
                    "To": packet['destination'],
                    "Queue Size": self.bundle_store.qsize()
                })
            else:
                logger.log_event("Invalid Payload", {
                    "Payload": packet['payload']
                }, is_error=True)

    def store_packet(self, destination, data, ttl=300):
        """Store a packet/bundle for future delivery"""
        packet = Packet(self.name, destination, PacketType.DATA)
        packet.payload = data
        packet.sequence_number = self.sequence_counter
        self.sequence_counter += 1
        
        # Store in queue
        self.bundle_store.put({
            'packet': packet,
            'stored_at': time.time(),
            'ttl': ttl
        })
        
        logger.log_event("Packet Storage", {
            "Packet ID": packet.packet_id,
            "From": self.name,
            "To": destination,
            "Queue Size": self.bundle_store.qsize()
        })
        return packet.packet_id

    def _check_bundles(self, neighbor, crdt=False):
        """Check if any bundles can be delivered via this neighbor"""
        current_time = time.time()
        temp_queue = Queue()
        forwarded_count = 0
        
        while not self.bundle_store.empty():
            bundle = self.bundle_store.get()
            packet = bundle['packet']
            stored_time = bundle['stored_at']
            ttl = bundle['ttl']
            
            packet_id = packet.get('packet_id') if isinstance(packet, dict) else packet.packet_id
            destination = packet.get('destination') if isinstance(packet, dict) else packet.destination
            
            if current_time - stored_time > ttl:
                logger.log_event("Expired Bundle", {
                    "Node": self.name,
                    "Packet ID": packet_id,
                    "Stored Time": f"{stored_time:.2f}s",
                    "TTL": ttl
                })
                continue
            
            should_forward = False
            if crdt:
                should_forward = is_good_next_hop(self, neighbor, destination, self.network_state)
                if forwarded_count >= 5:
                    should_forward = False
            else:
                should_forward = True
            
            if should_forward and self.packets_sent < self.max_packets:
                if send_packet(packet, neighbor.port):
                    self.packets_sent += 1
                    forwarded_count += 1
                    self.forwarded_packets.add(packet_id)
                    logger.log_event("Packet Forwarded", {
                        "Node": self.name,
                        "Packet ID": packet_id,
                        "To": destination,
                        "Via": neighbor.name,
                        "CRDT Mode": crdt
                    })
                else:
                    temp_queue.put(bundle)
            else:
                temp_queue.put(bundle)
        
        self.bundle_store = temp_queue
        
        logger.log_event("Bundle Check Summary", {
            "Node": self.name,
            "Neighbor": neighbor.name,
            "Packets Forwarded": forwarded_count,
            "Queue Size": self.bundle_store.qsize(),
            "CRDT Mode": crdt
        })

    def _find_next_hop(self, destination):
        """Find next hop for a destination based on contact history"""
        try:
            # If destination is a direct neighbor, return it
            if destination in [contact['node'] for contact in self.contact_history]:
                return get_node_by_name(destination)
            
            # Otherwise, return the current neighbor we're encountering
            # This simplifies routing for this example
            return get_node_by_name(destination)
        
        except Exception as e:
            error(f"u: {str(e)}\n")
            return None

    def _generate_and_send_packet(self, neighbor):
        """Generate a packet with random destination and try to send it"""
        try:
            # Get all possible destinations (excluding self and immediate neighbor)
            all_nodes = list(NODE_REGISTRY.keys())
            possible_destinations = [name for name in all_nodes 
                               if name != self.name and name != neighbor.name]
            
            if not possible_destinations:
                logger.log_event("No Other Destinations", {
                    "Node": self.name
                })
                return
            
            # Select random destination
            destination = random.choice(possible_destinations)
            
            # Generate packet data
            data = {
                'message': f"Hello from {self.name} at {time.strftime('%H:%M:%S')}",
                'type': 'encounter_generated',
                'encounter_with': neighbor.name
            }
            
            logger.log_event("Packet Generation", {
                "Source": self.name,
                "Destination": destination,
                "Via": neighbor.name,
                "Status": 'Stored' if self.packets_sent >= self.max_packets else 'Sent',
                "Packets": f"{self.packets_sent}/{self.max_packets}"
            })
            
            # Try to send the packet
            if self.packets_sent >= self.max_packets:
                # If we can't send, store in bundle store
                bundle_id = self.store_packet(destination, data)
                logger.log_event("Packet Stored", {
                    "Node": self.name,
                    "Packet ID": bundle_id,
                    "Destination": destination
                })
            else:
                # Send directly to neighbor
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    packet = Packet(self.name, destination, PacketType.DATA)
                    packet.payload = data
                    packet.sequence_number = self.sequence_counter
                    self.sequence_counter += 1
                    
                    # Convert to JSON-serializable dict
                    packet_dict = packet.to_dict()
                    packet_json = json.dumps(packet_dict)
                    sock.sendto(packet_json.encode(), ('127.0.0.1', neighbor.port))
                    
                    self.packets_sent += 1
                    self.forwarded_packets.add(packet.packet_id)
                    logger.log_event("Packet Sent", {
                        "Node": self.name,
                        "Packet ID": packet.packet_id,
                        "Destination": destination,
                        "Via": neighbor.name
                    })
                finally:
                    sock.close()
                
        except Exception as e:
            logger.log_event("Packet Generation Error", {
                "Error": str(e)
            }, is_error=True)

    def send_crdt_update(self, destination):
        """Send CRDT state to another node"""
        try:
            crdt_data = {
                'network_state': self.network_state.to_dict(),
                'forwarded_packets': self.forwarded_packets.elements,
                'timestamp': time.time()
            }
            
            dest_node = get_node_by_name(destination)
            if not dest_node:
                error(f"Cannot find node: {destination}\n")
                return False

            # Try to send the packet
            if self.crdt_packets_sent >= self.max_crdt_packets:
                logger.log_event("CRDT Packets Full", {
                    "Node": self.name,
                    "CRDT Packets": f"{self.crdt_packets_sent}/{self.max_crdt_packets}"
                })
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    packet = Packet(self.name, destination, PacketType.CRDT)
                    packet.payload = crdt_data
                    packet.sequence_number = self.sequence_counter
                    self.sequence_counter += 1
                    
                    # Convert to JSON-serializable dict
                    packet_dict = packet.to_dict()
                    packet_json = json.dumps(packet_dict)
                    sock.sendto(packet_json.encode(), ('127.0.0.1', dest_node.port))
                    
                    self.crdt_packets_sent += 1
                    logger.log_event("CRDT Update", {
                        "Source": self.name,
                        "Destination": destination,
                        "Status": "Success"
                    })
                    return True
                finally:
                    sock.close()
            
        except Exception as e:
            logger.log_event("CRDT Update Error", {
                "Error": str(e)
            }, is_error=True)
            return False
    
    def update_network_state(self, neighbor, rssi):
        """Record a contact with another node"""
        self.network_state.update_node(
            self.name,
            position=self.position if hasattr(self, 'position') else (0,0,0),
            transmission_range=100.0,  
            battery_level=self.battery_level if hasattr(self, 'battery_level') else 100.0,
            rssi=rssi,
            forwarded_packets=self.forwarded_packets.elements
        )
        
        # Record encounter with neighbor
        neighbor_info = NodeInfo(
            position=neighbor.position if hasattr(neighbor, 'position') else (0,0,0),
            transmission_range=100.0,
            battery_level=neighbor.battery_level if hasattr(neighbor, 'battery_level') else 100.0,
            last_update=time.time(),
            rssi=rssi,
            forwarded_packets=neighbor.forwarded_packets.elements
        )
        
        self.network_state.update_neighbor(
            self.name,
            neighbor.name,
            rssi,
            self.position if hasattr(self, 'position') else (0,0,0),
            neighbor_info
        )
 