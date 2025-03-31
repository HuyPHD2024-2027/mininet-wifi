from enum import Enum
import time
import json
import hashlib 

class PacketType(Enum):
    CRDT = 1
    DATA = 2
    
    def to_json(self):
        """Make PacketType JSON serializable"""
        return self.value

class Packet:
    """Network packet that mimics TCP/UDP structure"""
    def __init__(self, source, destination, packet_type=PacketType.DATA, ttl=60):
        self.source = source
        self.destination = destination
        self.packet_type = packet_type
        self.ttl = ttl
        self.timestamp = time.time()
        self.payload = None
        self.sequence_number = 0
        self.hop_count = 0
        self.packet_id = self._generate_packet_id()
    
    def increase_hop_count(self):
        self.hop_count += 1
        
    def _generate_packet_id(self):
        """Generate a unique packet ID"""
        # Combine source, timestamp, and sequence number for uniqueness
        id_string = f"{self.source}-{self.timestamp}-{self.sequence_number}"
        # Create a short hash for the ID
        return hashlib.md5(id_string.encode()).hexdigest()[:8]
        
    def __str__(self):
        """String representation of the packet"""
        return (f"Packet[{self.packet_type.name}] from {self.source} to {self.destination}, "
                f"seq={self.sequence_number}, hop_count={self.hop_count}")
        
    def to_dict(self):
        """Convert packet to dictionary, ensuring all values are JSON serializable"""
        return {
            'source': self.source,
            'destination': self.destination,
            'packet_type': self.packet_type.name,  # Use name instead of enum
            'ttl': self.ttl,
            'timestamp': self.timestamp,
            'payload': self.payload,
            'sequence_number': self.sequence_number,
            'packet_id': self.packet_id,
            'hop_count': self.hop_count
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create packet from dictionary, converting string back to enum"""
        packet = cls(data['source'], data['destination'])
        packet.packet_type = PacketType[data['packet_type']]  # Convert name back to enum
        packet.ttl = data['ttl']
        packet.timestamp = data['timestamp']
        packet.payload = data['payload']
        packet.sequence_number = data['sequence_number']
        packet.packet_id = data['packet_id']
        return packet

class CRDTPacket(Packet):
    """Specialized packet for CRDT operations"""
    def __init__(self, source, destination, crdt_data, ttl=64):
        super().__init__(source, destination, PacketType.CRDT, ttl)
        self.payload = crdt_data
    
    def __str__(self):
        base_str = super().__str__()
        crdt_summary = self.get_crdt_summary()
        return f"{base_str}\n  CRDT Details: {crdt_summary}"
    
    def get_crdt_summary(self):
        """Generate a summary of CRDT data"""
        if not isinstance(self.payload, dict):
            return "Invalid CRDT data"
        
        summary = []
        if 'bundle_store' in self.payload:
            bundle_count = len(self.payload['bundle_store'])
            summary.append(f"Bundles: {bundle_count}")
        if 'network_state' in self.payload:
            counter_val = sum(self.payload['network_state'].values())
            summary.append(f"Counter: {counter_val}")
        if 'forwarded_packets' in self.payload:
            fwd_count = len(self.payload['forwarded_packets'])
            summary.append(f"Forwarded: {fwd_count}")
        
        return ", ".join(summary) 