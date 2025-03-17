"""Packet handling utilities"""
from mininet.log import info, error
import json
import socket
from mn_wifi.packet import Packet, PacketType
import time

class PacketHandler:
    def process_incoming_packet(self, packet_dict):
        """Route incoming packet to appropriate handler"""
        if packet_dict['packet_type'] == "CRDT":
            self._process_crdt_sync(packet_dict)
        elif packet_dict['packet_type'] == "DATA":
            self._process_data_message(packet_dict)
    
    def _process_crdt_sync(self, packet):
        """Process incoming CRDT synchronization packet"""
        try:
            if not packet['payload'] or not isinstance(packet['payload'], dict):
                error(f"Invalid CRDT payload from {packet['source']}\n")
                return
            
            crdt_data = packet['payload']
            
            # Merge network state
            if 'network_state' in crdt_data:
                other_counter = GCounter.from_dict(crdt_data['network_state'])
                self.network_state.merge(other_counter)
            
            # Merge forwarded packets set
            if 'forwarded_packets' in crdt_data:
                other_packets = OrSet()
                other_packets.elements = crdt_data['forwarded_packets']
                self.forwarded_packets.merge(other_packets)
            
            info(f"*** {self.name} merged CRDT data from {packet['source']}\n")
                
        except Exception as e:
            error(f"Error handling CRDT packet: {str(e)}\n")
            import traceback
            error(traceback.format_exc())
    
    def _process_data_message(self, packet):
        """Process incoming data message packet"""
        print(f"*** {self.name} received DATA packet {packet['packet_id']} from {packet['source']} to {packet['destination']}\n")
        
        # If we're the destination
        if packet['destination'] == self.name and packet['destination'] != 'broadcast':
            print(f"*** Packet {packet['packet_id']} from {packet['source']} reached destination {packet['destination']}\n")
            packet['delay'] = time.time() - packet['timestamp']
            return
            
        # Otherwise store for forwarding
        if isinstance(packet['payload'], dict):
            self.bundle_store.put({
                'packet': packet,
                'stored_at': time.time(),
                'ttl': 300  # 5 minutes TTL default
            })
            info(f"*** {self.name} stored DATA packet {packet['packet_id']} for forwarding "
                 f"(bundle store size: {self.bundle_store.qsize()})\n")
        else:
            info(f"*** {self.name} received non-dict payload: {packet['payload']}\n") 