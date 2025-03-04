#!/usr/bin/python

from mn_wifi.node import Station
from mn_wifi.crdt import GCounter, OrSet
from mn_wifi.packet import Packet, PacketType
from mn_wifi.constants import NODE_REGISTRY, register_node, get_node_by_name
from mininet.log import info, error
import time
import json
import random
from threading import Thread, Lock
import socket
from socket import SO_REUSEADDR, SOL_SOCKET

class OpportunisticNode(Station):
    """Station with opportunistic networking capabilities"""
    
    # Class variable for port management
    _used_ports = set()
    _port_lock = Lock()
    
    @classmethod
    def _get_available_port(cls, start_port=9000, end_port=9999):
        """Get an available port number"""
        with cls._port_lock:
            available_ports = set(range(start_port, end_port + 1)) - cls._used_ports
            if not available_ports:
                raise RuntimeError("No available ports")
            port = random.choice(list(available_ports))
            cls._used_ports.add(port)
            return port
    
    def __init__(self, name, **params):
        super(OpportunisticNode, self).__init__(name, **params)
        
        # Register this node
        register_node(name, self)
        
        # CRDT data structures
        self.bundle_store = {}  # Store for pending packets/bundles
        self.rl_states_counter = GCounter()
        self.forwarded_packets = OrSet()
        
        # Network settings
        self.port = params.get('port') or self._get_available_port()
        self.max_packets = params.get('max_packets', 1)
        self.packets_sent = 0
        self.sequence_counter = 0
        
        # Tracking
        self.contact_history = []
        self.received_packets = set()
        
        # Start packet listener
        self.listening = True
        self.listener_thread = Thread(target=self._packet_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
        info(f"*** {self.name} initialized with port {self.port}, max packets: {self.max_packets}\n")
    
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
        """Process received packet"""
        if packet['packet_id'] in self.received_packets:
            return
            
        self.received_packets.add(packet['packet_id'])
        info(f"*** {self.name} received {packet}\n")
        
        # Process based on type
        if packet['packet_type'] == PacketType.CRDT:
            self._handle_crdt_packet(packet)
        elif packet['packet_type'] == PacketType.DATA:
            self._handle_data_packet(packet)
    
    def _handle_crdt_packet(self, packet):
        """Handle received CRDT packet"""
        try:
            if not packet['payload'] or not isinstance(packet['payload'], dict):
                error(f"Invalid CRDT payload from {packet['source']}\n")
                return
            
            crdt_data = packet['payload']
            
            # Merge bundle stores
            if 'bundle_store' in crdt_data:
                bundles_merged = 0
                for bundle_id, bundle_dict in crdt_data['bundle_store'].items():
                    if bundle_id not in self.bundle_store:
                        # Convert dict back to Packet object
                        packet = Packet.from_dict(bundle_dict['packet'])
                        self.bundle_store[bundle_id] = {
                            'packet': packet,
                            'stored_at': bundle_dict['stored_at'],
                            'ttl': bundle_dict['ttl']
                        }
                        bundles_merged += 1
            
            # Merge counter
            if 'rl_states_counter' in crdt_data:
                other_counter = GCounter()
                other_counter.counters = crdt_data['rl_states_counter']
                self.rl_states_counter.merge(other_counter)
            
            # Merge forwarded packets
            if 'forwarded_packets' in crdt_data:
                other_packets = OrSet()
                other_packets.elements = crdt_data['forwarded_packets']
                self.forwarded_packets.merge(other_packets)
            
            info(f"*** {self.name} merged CRDT data from {packet['source']}\n")
        
        except Exception as e:
            error(f"Error handling CRDT packet: {str(e)}\n")
            import traceback
            error(traceback.format_exc())
    
    def _handle_data_packet(self, packet):
        """Handle data packet"""
        # Check if packet reach destination
        
        if packet.destination == self.name and packet.destination != 'broadcast':
            print(f"*** Packet {packet.packet_id} from {packet.source} reached destination {packet.destination}\n")
            packet.delay = time.time() - packet.timestamp
        else:
            info(f"*** {self.name} received packet: {packet.packet_id}\n")
        
        # Store data
        if isinstance(packet.payload, dict) and 'id' in packet.payload:
            self.bundle_store[packet.packet_id] = packet.payload
    
    def store_bundle(self, destination, data, ttl=300):
        """Store a packet/bundle for future delivery"""
        packet = Packet(self.name, destination, PacketType.DATA)
        packet.payload = data
        packet.sequence_number = self.sequence_counter
        self.sequence_counter += 1
        
        self.bundle_store[packet.packet_id] = {
            'packet': packet,
            'stored_at': time.time(),
            'ttl': ttl
        }
        
        info(f"*** {self.name} stored packet {packet.packet_id} for {destination}\n")
        return packet.packet_id

    def _check_bundles(self, neighbor):
        """Check if any bundles can be delivered via this neighbor"""
        current_time = time.time()
        to_remove = []
        
        for bundle_id, bundle in self.bundle_store.items():
            packet = bundle['packet']
            stored_time = bundle['stored_at']
            ttl = bundle['ttl']
            
            # Check if bundle has expired
            if current_time - stored_time > ttl:
                to_remove.append(bundle_id)
                info(f"*** {self.name} removing expired bundle {bundle_id}\n")
                continue
            
            # If neighbor is destination or might lead to destination
            if neighbor.name == packet.destination or self._is_good_next_hop(neighbor, packet.destination):
                if self.packets_sent < self.max_packets:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    try:
                        packet_json = json.dumps(packet.to_dict())
                        sock.sendto(packet_json.encode(), ('127.0.0.1', neighbor.port))
                        
                        self.packets_sent += 1
                        info(f"*** {self.name} forwarded bundle {bundle_id} to {neighbor.name}\n")
                        to_remove.append(bundle_id)
                    finally:
                        sock.close()
        
        # Remove delivered or expired bundles
        for bundle_id in to_remove:
            del self.bundle_store[bundle_id]

    def handle_encounter(self, neighbor, rssi):
        """Handle encountering another node"""
        try:
            # Update contact history
            self.update_contact_history(neighbor.name, rssi)
            
            # 1. Generate and send a packet with arbitrary destination
            self._generate_and_send_random_packet(neighbor)
            
            # 2. Exchange CRDT data
            self.send_crdt_update(neighbor.name)
            
            # # Check bundles that might be deliverable
            # self._check_bundles(neighbor)
            
            info(f"*** {self.name} completed encounter with {neighbor.name}\n")
        
        except Exception as e:
            error(f"Error in handle_encounter: {str(e)}\n")
            import traceback
            error(traceback.format_exc())

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
            error(f"Error finding next hop: {str(e)}\n")
            return None

    def _generate_and_send_random_packet(self, neighbor):
        """Generate a packet with random destination and try to send it"""
        try:
            # Get all possible destinations (excluding self and immediate neighbor)
            all_nodes = list(NODE_REGISTRY.keys())
            possible_destinations = [name for name in all_nodes 
                               if name != self.name and name != neighbor.name]
            
            if not possible_destinations:
                info(f"*** {self.name} no other destinations available\n")
                return
            
            # Select random destination
            destination = random.choice(possible_destinations)
            
            # Generate packet data
            data = {
                'message': f"Hello from {self.name} at {time.strftime('%H:%M:%S')}",
                'type': 'encounter_generated',
                'encounter_with': neighbor.name
            }
            
            info(f"\n*** {self.name} generating random packet for {destination} "
                 f"during encounter with {neighbor.name}\n")
            
            # Try to send the packet
            if self.packets_sent >= self.max_packets:
                # If we can't send, store in bundle store
                bundle_id = self.store_bundle(destination, data)
                info(f"*** {self.name} stored generated packet {bundle_id} "
                     f"(max packets reached: {self.packets_sent}/{self.max_packets})\n")
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
                    info(f"*** {self.name} sent generated packet {packet.packet_id} "
                         f"({self.packets_sent}/{self.max_packets}) for {destination} via {neighbor.name}\n")
                finally:
                    sock.close()
                
        except Exception as e:
            error(f"Error generating random packet: {str(e)}\n")
            import traceback
            error(traceback.format_exc())

    def send_crdt_update(self, destination):
        """Send CRDT state to another node"""
        try:
            # # Prepare CRDT data
            # bundle_store_dict = {}
            # for k, v in self.bundle_store.items():
            #     packet_dict = v['packet'].to_dict()  # Convert packet to dict
            #     bundle_store_dict[k] = {
            #         'packet': packet_dict,
            #         'stored_at': v['stored_at'],
            #         'ttl': v['ttl']
            #     }
            
            crdt_data = {
                # 'bundle_store': bundle_store_dict,
                'rl_states_counter': self.rl_states_counter.counters,
                'forwarded_packets': self.forwarded_packets.elements,
                'timestamp': time.time()
            }
            
            dest_node = get_node_by_name(destination)
            if not dest_node:
                error(f"Cannot find node: {destination}\n")
                return False
            
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
                
                info(f"*** {self.name} sent CRDT update to {destination} "
                     f"(bundles: {len(self.bundle_store)})\n")
                return True
            finally:
                sock.close()
            
        except Exception as e:
            error(f"Error sending CRDT update: {str(e)}\n")
            import traceback
            error(traceback.format_exc())
            return False
    
    def update_contact_history(self, node_name, rssi):
        """Record a contact with another node"""
        contact = {
            'node': node_name,
            'rssi': rssi,
            'timestamp': time.time(),
            'position': self.position if hasattr(self, 'position') else None
        }
        
        self.contact_history.append(contact)
        # Keep only recent contacts
        if len(self.contact_history) > 50:
            self.contact_history = self.contact_history[-50:]
    
    def send_ack(self, packet):
        """Send acknowledgment for received packet"""
        ack = Packet(self.name, packet.source, packet.packet_type)
        ack.acknowledgment = True
        ack.sequence_number = packet.sequence_number
        self.send_packet_to(packet.source, None, packet.packet_type)
    
    def send_packet_to(self, destination, data, packet_type=PacketType.DATA):
        """Send packet to an arbitrary destination"""
        if self.packets_sent >= self.max_packets:
            info(f"*** {self.name} reached packet limit ({self.packets_sent}/{self.max_packets}), "
                 f"storing packet for {destination}\n")
            self.store_bundle(destination, data)
            return False
            
        try:
            packet = Packet(self.name, destination, packet_type)
            packet.payload = data
            packet.sequence_number = self.sequence_counter
            self.sequence_counter += 1
            
            info(f"*** {self.name} creating packet {packet.packet_id} for {destination}\n")
            
            # Try to find direct neighbor first
            next_hop = self._find_next_hop(destination)
            if next_hop:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    packet_json = json.dumps(packet.to_dict())
                    sock.sendto(packet_json.encode(), ('127.0.0.1', next_hop.port))
                    
                    self.packets_sent += 1
                    info(f"*** {self.name} sent packet {packet.packet_id} "
                         f"({self.packets_sent}/{self.max_packets}) towards {destination} via {next_hop.name}\n")
                    return True
                finally:
                    sock.close()
            else:
                # Store in bundle_store for opportunistic delivery
                self.bundle_store[packet.packet_id] = {
                    'packet': packet,
                    'stored_at': time.time(),
                    'ttl': 300  # 5 minutes TTL default
                }
                info(f"*** {self.name} stored packet {packet.packet_id} in bundle store for future delivery\n")
                return True
                
        except Exception as e:
            error(f"Error sending packet: {str(e)}\n")
            return False