#!/usr/bin/python

from mn_wifi.node import Station
from mn_wifi.crdt import GCounter, OrSet, NodeInfo
from mn_wifi.packet import Packet, PacketType
from mn_wifi.constants import NODE_REGISTRY, register_node, get_node_by_name
from mininet.log import info, error
import time
import json
import random
from threading import Thread, Lock
import socket
from socket import SO_REUSEADDR, SOL_SOCKET
from queue import Queue

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
        self.bundle_store = Queue()  # Change to Queue
        self.network_state = GCounter()
        self.forwarded_packets = OrSet()
        
        # Network settings
        self.port = params.get('port') or self._get_available_port()
        self.max_packets = params.get('max_packets', 1)
        self.max_crdt_packets = params.get('max_crdt_packets', 1)
        
        self.packets_sent = 0
        self.crdt_packets_sent = 0
        self.sequence_counter = 0
        
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
        """Process received packet based on type"""
        if packet['packet_type'] == "CRDT":
            self._handle_crdt_packet(packet)
        elif packet['packet_type'] == "DATA":
            self._handle_data_packet(packet)
    
    def _handle_crdt_packet(self, packet):
        """Handle received CRDT packet"""
        try:
            if not packet['payload'] or not isinstance(packet['payload'], dict):
                error(f"Invalid CRDT payload from {packet['source']}\n")
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
            
            info(f"*** {self.name} merged CRDT data from {packet['source']}\n")
            # info(f"Network state: {self.network_state.to_dict()}\n")
            # info(f"Received packets: {self.forwarded_packets.elements}\n")
                
        except Exception as e:
            error(f"Error handling CRDT packet: {str(e)}\n")
            import traceback
            error(traceback.format_exc())
    
    def _handle_data_packet(self, packet):
        """Handle data packet"""
        # Check if packet reach destination
        print(f"*** {self.name} received DATA packet {packet['packet_id']} from {packet['source']} to {packet['destination']}\n")
        if packet['destination'] == self.name and packet['destination'] != 'broadcast':
            print(f"*** Packet {packet['packet_id']} from {packet['source']} reached destination {packet['destination']}\n")
            packet['delay'] = time.time() - packet['timestamp']
        else:
            # Store data in queue
            if isinstance(packet['payload'], dict):
                self.bundle_store.put({
                    'packet': packet,
                    'stored_at': time.time(),
                    'ttl': 300  # 5 minutes TTL default
                })
                info(f"*** {self.name} receive and add DATA packet {packet['packet_id']} to bundle store "
                f"size: {self.bundle_store.qsize()}\n")
            else:
                info(f"*** {self.name} received non-dict payload: {packet['payload']}\n")

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
        
        info(f"*** {self.name} stored packet {packet.packet_id} for {destination}\n")
        return packet.packet_id

    def _is_good_next_hop(self, neighbor, destination):
        """Determine if a neighbor is a good next hop for a destination"""
        # If crdt data is not available, just return True as default
        if not hasattr(self, 'network_state') or not self.network_state:
            return True
            
        # Get destination node's last known position
        dest_position = None
        dest_node = get_node_by_name(destination)
        if dest_node and hasattr(dest_node, 'position'):
            dest_position = dest_node.position
        
        # If destination node is unknown or position not available,
        # check if neighbor has encountered destination before
        for node_id, state in self.network_state.states.items():
            if destination in state.encounter_history:
                # Neighbor has encountered destination before
                return True
                
        # Calculate metrics for forwarding decision
        neighbor_metrics = self._calculate_neighbor_metrics(neighbor, destination, dest_position)
        
        # Simple decision: if metrics score > 0.5, consider it a good next hop
        print(f"*** {self.name} neighbor {neighbor.name} metrics: {neighbor_metrics}")
        return neighbor_metrics > 0.2
    
    def _calculate_neighbor_metrics(self, neighbor, destination, dest_position=None):
        """Calculate neighbor metrics for forwarding decision"""
        metrics_score = 0.5  # Default score
        
        # Check if we have state for this neighbor
        if neighbor.name in self.network_state.states:
            state = self.network_state.states[neighbor.name]
            
            # 1. Check link quality to this neighbor
            if neighbor.name in state.link_quality:
                link_quality = state.link_quality[neighbor.name]
                metrics_score += link_quality * 0.2  # Weight link quality by 0.2
            
            # 2. Check if neighbor has encountered destination
            if destination in state.encounter_history:
                encounter = state.encounter_history[destination]
                
                # More recent encounters are weighted higher
                recency = 1.0 - min(1.0, (time.time() - encounter['timestamp']) / (60 * 60))  # Scale by hour
                metrics_score += recency * 0.3  # Weight recency by 0.3
                
                # If we know destination position, calculate distance
                if dest_position and 'position' in encounter:
                    distance = self._calculate_distance(dest_position, encounter['position'])
                    proximity = 1.0 - min(1.0, distance / 1000)  # Scale by 1000 meters
                    metrics_score += proximity * 0.2  # Weight proximity by 0.2
            
            # 3. Check congestion level (lower is better)
            congestion_factor = 1.0 - state.congestion_level
            metrics_score += congestion_factor * 0.1  # Weight congestion by 0.1
            
            # 4. Check battery level (higher is better)
            if state.node_info and state.node_info.battery_level:
                battery_factor = state.node_info.battery_level / 100.0
                metrics_score += battery_factor * 0.1  # Weight battery by 0.1
                
        return metrics_score
        
    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5

    def _check_bundles(self, neighbor, crdt=False):
        """Check if any bundles can be delivered via this neighbor"""
        current_time = time.time()
        temp_queue = Queue()  # Temporary queue for storing undelivered packets
        forwarded_count = 0
        
        # Process all packets in queue
        while not self.bundle_store.empty():
            bundle = self.bundle_store.get()
            packet = bundle['packet']
            stored_time = bundle['stored_at']
            ttl = bundle['ttl']
            
            # Get packet_id and destination based on packet type
            packet_id = packet.get('packet_id') if isinstance(packet, dict) else packet.packet_id
            destination = packet.get('destination') if isinstance(packet, dict) else packet.destination
            
            # # Skip if packet is already in forwarded_packets
            # if crdt and packet_id in self.forwarded_packets:
            #     info(f"*** {self.name} skipping bundle {packet_id} (already forwarded/delivered)\n")
            #     continue
                
            # Check if bundle has expired
            if current_time - stored_time > ttl:
                info(f"*** {self.name} removing expired bundle {packet_id}\n")
                continue
            
            # Enhanced forwarding decision when CRDT is enabled
            should_forward = False
            if crdt:
                # Determine if this neighbor is a good next hop based on network state
                should_forward = self._is_good_next_hop(neighbor, destination)
                
                # Check if we're at maximum forwards per encounter
                if forwarded_count >= 5:  # Limit to 5 forwards per encounter to prevent flooding
                    should_forward = False
            else:
                # Without CRDT, use simple forwarding
                should_forward = True
            
            # Forward packet if conditions are met
            if should_forward and self.packets_sent < self.max_packets:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # Convert packet to dictionary or use packet as is if it's already a dict
                    if hasattr(packet, 'to_dict'):
                        packet_data = packet.to_dict()
                    else:
                        packet_data = packet
                        
                    packet_json = json.dumps(packet_data)
                    sock.sendto(packet_json.encode(), ('127.0.0.1', neighbor.port))
                    
                    self.packets_sent += 1
                    forwarded_count += 1
                    
                    # Add to forwarded_packets to mark as forwarded
                    self.forwarded_packets.add(packet_id)
                    
                    info(f"*** {self.name} forwarded packet {packet_id} to {neighbor.name} "
                         f"using {'CRDT-enhanced' if crdt else 'simple'} routing\n")
                except Exception as e:
                    error(f"Error forwarding packet: {str(e)}\n")
                    # Put back in queue if forwarding failed
                    temp_queue.put(bundle)
                finally:
                    sock.close()
            else:
                # Put back in queue if not forwarded
                temp_queue.put(bundle)
        
        # Restore undelivered packets to bundle_store
        self.bundle_store = temp_queue
        
        if crdt:
            info(f"*** {self.name} completed bundle check with {neighbor.name}, "
                 f"forwarded {forwarded_count} packets, {self.bundle_store.qsize()} remaining in queue\n")

    def handle_encounter(self, neighbor, rssi, crdt=False):
        """Handle encountering another node"""
        try:
            # Update contact history
            self.update_network_state(neighbor, rssi)
            
            # 1. Generate and send a packet with arbitrary destination
            self._generate_and_send_packet(neighbor)
            
            # 2. Exchange CRDT data
            if crdt:
                self.send_crdt_update(neighbor.name)

            # 3. Check bundles that might be deliverable
            self._check_bundles(neighbor, crdt)

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
            
            info(f"\n*** {self.name} generating DATA packet for {destination} "
                 f"during encounter with {neighbor.name}\n")
            
            # Try to send the packet
            if self.packets_sent >= self.max_packets:
                # If we can't send, store in bundle store
                bundle_id = self.store_packet(destination, data)
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
                    self.forwarded_packets.add(packet.packet_id)
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
                info(f"(max crdt packets reached: {self.crdt_packets_sent}/{self.max_crdt_packets})\n")
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
                    info(f"*** {self.name} sent CRDT update to {destination}\n")
                    return True
                finally:
                    sock.close()
            
        except Exception as e:
            error(f"Error sending CRDT update: {str(e)}\n")
            import traceback
            error(traceback.format_exc())
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
            self.store_packet(destination, data)
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
                # Store in bundle_store queue for opportunistic delivery
                self.bundle_store.put({
                    'packet': packet,
                    'stored_at': time.time(),
                    'ttl': 300  # 5 minutes TTL default
                })
                info(f"*** {self.name} stored packet {packet.packet_id} in bundle store for future delivery\n")
                return True
                
        except Exception as e:
            error(f"Error sending packet: {str(e)}\n")
            return False
            return False
            return False