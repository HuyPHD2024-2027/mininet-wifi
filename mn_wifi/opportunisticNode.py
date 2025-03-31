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
from collections import deque

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
        self.ttl = params.get('ttl', 60)
        # Set to float('inf') for unlimited packets
        self.max_packets = params.get('max_packets', float('inf'))
        self.max_crdt_packets = params.get('max_crdt_packets', float('inf'))
        
        self.packets_sent = 0
        self.crdt_packets_sent = 0
        self.sequence_counter = 0
        
        # Add throughput tracking variables
        self.delivered_packets = 0
        self.start_time = time.time()
        self.throughput_window = params.get('throughput_window', 10)
        self.packet_history = deque(maxlen=1000)
        self.bytes_delivered = 0  # Track total bytes delivered
        
        # Packet generation settings
        self.packet_gen_interval = params.get('packet_gen_interval', 5)  # Default to 5 seconds
        
        # Start packet listener
        self.listening = True
        self.listener_thread = Thread(target=self._packet_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
        # Start periodic packet generator
        self.packet_gen_thread = Thread(target=self._periodic_packet_generator)
        self.packet_gen_thread.daemon = True
        self.packet_gen_thread.start()
        
        logger.log_event("Node Initialization", {
            "Name": self.name,
            "Port": self.port,
            "Max Packets": "Unlimited" if self.max_packets == float('inf') else self.max_packets,
            "Max CRDT Packets": "Unlimited" if self.max_crdt_packets == float('inf') else self.max_crdt_packets,
            "Beacon Interval": f"{self.packet_gen_interval}s"
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
                    logger.log_event("Error", {
                        "Error": str(e)
                    }, is_error=True)
        finally:
            sock.close()
    
    def _process_packet(self, packet):
        """Process received packet based on type"""
        if packet['packet_type'] == "CRDT":
            self._handle_crdt_packet(packet)
        elif packet['packet_type'] == "DATA":
            self._handle_data_packet(packet)
    
    def _handle_crdt_packet(self, packet):
        """Handle received CRDT packet with the updated OR-Set implementation"""
        try:
            if not packet['payload']:
                logger.log_event("CRDT Error", {
                    "Source": packet['source'],
                    "Error": "Invalid payload"
                }, is_error=True)
                
                return
            
            crdt_data = packet['payload']
            
            # Store old state before merging to track changes
            old_network_state = {k: v for k, v in self.network_state.to_dict().items()}
            old_forwarded_packets = {
                'pending': self.forwarded_packets._pending.copy() if hasattr(self.forwarded_packets, '_pending') else [],
                'acknowledged': self.forwarded_packets._acknowledged.copy() if hasattr(self.forwarded_packets, '_acknowledged') else []
            }
            
            # Merge counter for network state
            if 'network_state' in crdt_data:
                other_counter = GCounter.from_dict(crdt_data['network_state'])
                self.network_state.merge(other_counter)
            
            # Merge forwarded packets set
            if 'forwarded_packets' in crdt_data:
                other_packets = OrSet.from_dict(crdt_data['forwarded_packets'])
                self.forwarded_packets.merge(other_packets)
            
            # Log CRDT sync results
            logger.log_event("CRDT Sync", {
                "Source": packet['source'],
                "Destination": self.name,
                "Status": "Success",
            })
            
            # # Check for unacknowledged packets that need retransmission
            # self._check_for_retransmissions(packet['source'])
                
        except Exception as e:
            logger.log_event("CRDT Error", {
                "Source": packet['source'],
                "Error": str(e)
            }, is_error=True)
    
    def _check_for_retransmissions(self, neighbor_name):
        """Check for packets that need retransmission to a neighbor based on OR-Set state"""
        try:
            # Get the neighbor node
            neighbor = get_node_by_name(neighbor_name)
            if not neighbor:
                logger.log_event("Retransmission Error", {
                    "Error": f"Cannot find node: {neighbor_name}"
                }, is_error=True)
                return
            
            # Check pending packets that might need retransmission
            pending_packets = self.forwarded_packets._pending
            if not pending_packets:
                return
            
            logger.log_event("Retransmission Check", {
                "Node": self.name,
                "Neighbor": neighbor_name,
                "Pending Packets": len(pending_packets)
            })
            
            # Create temporary bundles for pending packets that need retransmission
            for packet_id in pending_packets[:5]:  # Limit to 5 packets max
                # Create a retransmission packet
                packet = Packet(self.name, neighbor_name, PacketType.DATA)
                packet.payload = {
                    'message': f"Retransmission from {self.name}",
                    'type': 'retransmission',
                    'original_packet_id': packet_id
                }
                packet.sequence_number = self.sequence_counter
                self.sequence_counter += 1
                
                # Add to bundle store for delivery
                self.bundle_store.put({
                    'packet': packet,
                    'stored_at': time.time(),
                    'ttl': self.ttl,  
                    'retransmission': True
                })
                
                logger.log_event("Retransmission Added", {
                    "Node": self.name,
                    "Packet ID": packet_id,
                    "Destination": neighbor_name
                })
                
        except Exception as e:
            logger.log_event("Retransmission Error", {
                "Error": str(e)
            }, is_error=True)
    
    def _handle_data_packet(self, packet):
        """Handle data packet with enhanced throughput tracking and OR-Set updates"""
        packet_id = packet['packet_id']
        sender_id = packet['source']
        timestamp = packet['timestamp']
        packet['hop_count'] += 1

        if packet['destination'] == self.name and packet['destination'] != 'broadcast':
            current_time = time.time()
            delay = current_time - packet['timestamp']
            
            # Update packet statistics
            self.delivered_packets += 1
            self.packet_history.append(current_time)
            
            # Calculate packet size in bytes (approximate from JSON string)
            packet_size = len(json.dumps(packet).encode('utf-8'))
            self.bytes_delivered += packet_size
            
            # Mark packet as acknowledged in OR-Set
            self.forwarded_packets.remove(packet_id, sender_id, packet['timestamp'])
            
            # Calculate throughput metrics
            throughput_metrics = self._calculate_throughput_metrics()
            if hasattr(self, 'saver'):
                # Log packet delivery with throughput information
                self.saver.log_packet_delivery(
                    source=packet['source'],
                    destination=packet['destination'],
                    delay=delay,
                    success=True,
                    metrics={
                        'hop_count': packet['hop_count'],
                        'throughput': throughput_metrics['current_throughput'],
                        'packet_size': packet_size,
                        'cumulative_packets': self.delivered_packets,
                        'bytes_delivered': self.bytes_delivered
                    }
                )
                
                # Log periodic throughput statistics
                if self.delivered_packets % 10 == 0:  # Every 10 packets
                    self.saver.log_throughput_stats(
                        node_name=self.name,
                        stats={
                            'current_throughput': throughput_metrics['current_throughput'],
                            'average_throughput': throughput_metrics['average_throughput'],
                            'total_packets': throughput_metrics['total_packets'],
                            'running_time': throughput_metrics['running_time'],
                            'bytes_delivered': throughput_metrics['bytes_delivered'],
                            'bytes_throughput': throughput_metrics['bytes_throughput']
                        }
                    )
            
            logger.log_event("Packet Delivery", {
                "Packet ID": packet['packet_id'],
                "From": packet['source'],
                "To": packet['destination'],
                "Hop Count": packet['hop_count'],
                "Delay": f"{delay:.2f}s",
                "Throughput": f"{throughput_metrics['current_throughput']:.2f} packets/s",
                "Bytes": packet_size,
                "Total Delivered": self.delivered_packets,
                "Running Time": f"{throughput_metrics['running_time']:.2f}s"
            })
        else:
            # Store data in queue
            if isinstance(packet['payload'], dict):
                self.bundle_store.put({
                    'packet': packet,
                    'stored_at': time.time(),
                    'ttl': self.ttl
                })
                
                # Add to pending set in OR-Set
                self.forwarded_packets.add(packet_id, sender_id, timestamp)
                
                logger.log_event("Packet Storage", {
                    "Packet ID": packet['packet_id'],
                    "From": packet['source'],
                    "To": packet['destination'],
                    "Queue Size": self.bundle_store.qsize(),
                    "Pending Status": "Added to pending set"
                })
            else:
                logger.log_event("Invalid Payload", {
                    "Payload": packet['payload']
                }, is_error=True)

    def store_packet(self, destination, data):
        """Store a packet/bundle for future delivery"""
        packet = Packet(self.name, destination, PacketType.DATA)
        packet.payload = data
        packet.sequence_number = self.sequence_counter
        self.sequence_counter += 1
        
        # Store in queue
        self.bundle_store.put({
            'packet': packet,
            'stored_at': time.time(),
            'ttl': self.ttl
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
        acknowledged_count = 0
        
        while not self.bundle_store.empty():
            bundle = self.bundle_store.get()
            packet = bundle['packet']
            stored_time = bundle['stored_at']
            ttl = bundle['ttl']
            is_retransmission = bundle.get('retransmission', False)
            
            packet_id = packet.get('packet_id') if isinstance(packet, dict) else packet.packet_id
            destination = packet.get('destination') if isinstance(packet, dict) else packet.destination
            
            # Check for TTL expiration
            if current_time - stored_time > ttl:
                # Record a failed delivery attempt
                if hasattr(self, 'saver'):
                    self.saver.log_packet_expiry(
                        source=packet.get('source', self.name) if isinstance(packet, dict) else packet.source,
                        destination=destination,
                        packet_id=packet_id,
                        stored_time=stored_time,
                        ttl=ttl
                    )
                    
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
                if forwarded_count >= 5 and not is_retransmission:
                    should_forward = False
            else:
                should_forward = True
            
            if should_forward and self.packets_sent < self.max_packets:
                if send_packet(packet, neighbor.port):
                    self.packets_sent += 1
                    forwarded_count += 1
                    
                    # If this is a data packet being forwarded, mark it as acknowledged
                    if not is_retransmission and isinstance(packet, dict) and packet.get('packet_type') == 'DATA':
                        self.forwarded_packets.remove(packet_id, self.name, time.time())
                        acknowledged_count += 1
                    elif not is_retransmission:
                        sender_id = packet.source if hasattr(packet, 'source') else self.name
                        self.forwarded_packets.remove(packet_id, sender_id, time.time())
                        acknowledged_count += 1
                    
                    logger.log_event("Packet Forwarded", {
                        "Node": self.name,
                        "Packet ID": packet_id,
                        "To": destination,
                        "Via": neighbor.name,
                        "Hop Count": packet.get('hop_count') if isinstance(packet, dict) else packet.hop_count,
                        "CRDT Mode": crdt,
                        "Status": "Acknowledged" if not is_retransmission else "Retransmitted"
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
            "Packets Acknowledged": acknowledged_count,
            "Queue Size": self.bundle_store.qsize(),
            "CRDT Mode": crdt,
            "Pending Packets": len(self.forwarded_packets._pending),
            "Acknowledged Packets": len(self.forwarded_packets._acknowledged)
        })

    def send_crdt_update(self, destination):
        """Send CRDT state to another node"""
        try:
            # Convert forwarded_packets OrSet to dictionary for serialization
            forwarded_packets_dict = self.forwarded_packets.to_dict()
            
            crdt_data = {
                'network_state': self.network_state.to_dict(),
                'forwarded_packets': forwarded_packets_dict,
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
                        "Status": "Success",
                        "Pending Packets": len(self.forwarded_packets._pending),
                        "Acknowledged Packets": len(self.forwarded_packets._acknowledged)
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

    def _calculate_throughput_metrics(self):
        """Calculate comprehensive throughput metrics"""
        current_time = time.time()
        window_start = current_time - self.throughput_window
        
        # Count packets and bytes in the current window
        recent_packets = [p for p in self.packet_history if p > window_start]
        
        metrics = {
            'current_throughput': len(recent_packets) / self.throughput_window if recent_packets else 0,
            'average_throughput': self.delivered_packets / (current_time - self.start_time),
            'total_packets': self.delivered_packets,
            'running_time': current_time - self.start_time,
            'bytes_delivered': self.bytes_delivered,
            'bytes_throughput': self.bytes_delivered / (current_time - self.start_time)
        }
        
        return metrics

    def _periodic_packet_generator(self):
        """Periodically generate and send packets to all known neighbors"""
        # Wait for network initialization
        time.sleep(10)
        
        while self.listening:
            try:
                # Get neighbors from network state
                neighbors = []
                if self.name in self.network_state.states:
                    neighbors = list(self.network_state.states[self.name].neighbors.keys())
                
                if neighbors and self.packets_sent < self.max_packets:
                    logger.log_event("Packet Generation", {
                        "Node": self.name,
                        "Neighbors": neighbors,
                        "Total": len(neighbors)
                    })
                    
                    # Send packet to each neighbor
                    for neighbor_name in neighbors:
                        # Try to get neighbor node
                        neighbor = get_node_by_name(neighbor_name)
                        if not neighbor:
                            continue
                            
                        data = {
                            'message': f"Hello from {self.name} at {time.strftime('%H:%M:%S')}",
                            'type': 'periodic',
                        }
                        
                        # Create packet
                        packet = Packet(self.name, neighbor_name, PacketType.DATA)
                        packet.payload = data
                        packet.sequence_number = self.sequence_counter
                        self.sequence_counter += 1
                        
                        # Try to send packet directly
                        if send_packet(packet, neighbor.port):
                            self.packets_sent += 1
                            self.forwarded_packets.add(packet.packet_id, self.name, time.time())
                            
                            logger.log_event("Packet Sent", {
                                "Node": self.name,
                                "Destination": neighbor_name,
                                "Packet ID": packet.packet_id,
                                "Method": "direct"
                            })
                        else:
                            # If direct send fails, store for later delivery
                            self.bundle_store.put({
                                'packet': packet,
                                'stored_at': time.time(),
                                'ttl': self.ttl
                            })
                            
                            logger.log_event("Packet Stored", {
                                "Node": self.name,
                                "Destination": neighbor_name,
                                "Packet ID": packet.packet_id,
                                "Method": "bundled"
                            })

                                # If no neighbors found, try to get random destinations instead
                else:
                        logger.log_event("No Known Neighbors", {
                            "Node": self.name,
                            "Sleep for": "10 seconds"
                        })
                
                    
                # Wait for next interval
                time.sleep(self.packet_gen_interval)
                
            except Exception as e:
                logger.log_event("Packet Generation Error", {
                    "Node": self.name,
                    "Error": str(e)
                }, is_error=True)
                time.sleep(1)  # Brief delay before retrying
 