from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, field
from mininet.log import error

@dataclass
class NodeInfo:
    """Basic information about a node"""
    position: Tuple[float, float, float]  # (x, y, z)
    transmission_range: float
    battery_level: float
    last_update: float
    rssi: float = -100.0
    forwarded_packets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'position': self.position,
            'transmission_range': self.transmission_range,
            'battery_level': self.battery_level,
            'last_update': self.last_update,
            'rssi': self.rssi,
            'forwarded_packets': self.forwarded_packets
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NodeInfo':
        return cls(
            position=tuple(data['position']),
            transmission_range=data['transmission_range'],
            battery_level=data['battery_level'],
            last_update=data['last_update'],
            rssi=data.get('rssi', -100.0),
            forwarded_packets=data.get('forwarded_packets', [])
        )

class NetworkState:
    """Represents a node's network state with timestamp"""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.timestamp = time.time()
        self.MAX_HISTORY = 10  # Limit state history to 5 elements

        self.node_info: Optional[NodeInfo] = None
        self.neighbors: Dict[str, NodeInfo] = {}  # neighbor_id -> NodeInfo
        self.encounter_history: Dict[str, Dict] = {}  # neighbor_id -> most recent encounter

        self.packet_stats = {
            'sent': 0,
            'received': 0,
            'forwarded': 0,
            'dropped': 0
        }

        self.link_quality: Dict[str, float] = {}  # neighbor_id -> quality (0-1)
        self.congestion_level: float = 0.0  # 0-1 scale
        

    def update_node_info(self, position: Tuple[float, float, float], 
                        transmission_range: float, battery_level: float,
                        rssi: float = -100.0,
                        forwarded_packets: List[str] = []):
        """Update basic node information"""
        self.timestamp = time.time()
        self.node_info = NodeInfo(
            position=position,
            transmission_range=transmission_range,
            battery_level=battery_level,
            last_update=self.timestamp,
            rssi=rssi,
            forwarded_packets=forwarded_packets
        )
        self._record_state_change("node_info_update")

    def update_neighbor_info(self, neighbor_id: str, neighbor_info: NodeInfo):
        """Update information about a neighbor"""
        self.timestamp = time.time()
        self.neighbors[neighbor_id] = neighbor_info
        
        # Calculate and update link quality
        distance = self._calculate_distance(
            self.node_info.position if self.node_info else (0,0,0),
            neighbor_info.position
        )
        max_range = max(
            self.node_info.transmission_range if self.node_info else 0,
            neighbor_info.transmission_range
        )
        
        # Simple link quality metric based on distance and RSSI
        distance_quality = 1.0 - (distance / max_range if max_range > 0 else 1)
        rssi_quality = (neighbor_info.rssi + 100) / 100  # Normalize RSSI to 0-1
        self.link_quality[neighbor_id] = (distance_quality + rssi_quality) / 2
        
        self._record_state_change("neighbor_update", neighbor_id=neighbor_id)

    def record_encounter(self, neighbor_id: str, rssi: float, 
                        position: Tuple[float, float, float]):
        """Record an encounter with another node"""
        self.timestamp = time.time()
        encounter = {
            'neighbor_id': neighbor_id,
            'timestamp': self.timestamp,
            'rssi': rssi,
            'position': position,
            'link_quality': self.link_quality.get(neighbor_id, 0)
        }
        self.encounter_history[neighbor_id] = encounter  # Use dictionary assignment instead of append
        
        self._record_state_change("encounter", neighbor_id=neighbor_id)
    
    def update_packet_stats(self, action: str):
        """Update packet statistics"""
        self.timestamp = time.time()
        if action in self.packet_stats:
            self.packet_stats[action] += 1
        
        # Update congestion level based on packet stats
        total_packets = sum(self.packet_stats.values())
        if total_packets > 0:
            self.congestion_level = self.packet_stats['dropped'] / total_packets
        
        self._record_state_change("packet_stats", action=action)

    def _record_state_change(self, change_type: str, **extra_info):
        """Record a state change in history"""
        state_change = {
            'timestamp': self.timestamp,
            'type': change_type,
            **extra_info
        }
        
    def _calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5
        
    def to_dict(self) -> Dict:
        """Convert state to dictionary for serialization"""
        return {
            'node_id': self.node_id,
            'timestamp': self.timestamp,
            'node_info': self.node_info.to_dict() if self.node_info else None,
            'neighbors': {
                n_id: info.to_dict() 
                for n_id, info in self.neighbors.items()
            },
            'encounter_history': self.encounter_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NetworkState':
        """Create NetworkState from dictionary"""
        state = cls(data['node_id'])
        state.timestamp = data['timestamp']
        
        if data.get('node_info'):
            state.node_info = NodeInfo.from_dict(data['node_info'])
        
        state.neighbors = {
            n_id: NodeInfo.from_dict(n_info)
            for n_id, n_info in data.get('neighbors', {}).items()
        }
        
        state.encounter_history = data.get('encounter_history', {})
        state.packet_stats = data.get('packet_stats', {
            'sent': 0, 'received': 0, 'forwarded': 0, 'dropped': 0
        })
        state.link_quality = data.get('link_quality', {})
        state.congestion_level = data.get('congestion_level', 0.0)

        return state

class GCounter:
    """
    Grow-only Counter CRDT with enhanced network state tracking
    """
    def __init__(self):
        self.states: Dict[str, NetworkState] = {}
    
    def get_or_create_state(self, node_id: str) -> NetworkState:
        """Get existing state or create new one for node"""
        if node_id not in self.states:
            self.states[node_id] = NetworkState(node_id)
        return self.states[node_id]
    
    def update_node(self, node_id: str, position: Tuple[float, float, float],
                   transmission_range: float, battery_level: float,
                   rssi: float = -100.0,
                   forwarded_packets: List[str] = []):
        """Update node information"""
        state = self.get_or_create_state(node_id)
        state.update_node_info(position, transmission_range, battery_level, rssi, forwarded_packets)
    
    def update_neighbor(self, node_id: str, neighbor_id: str,
                        rssi: float, position: Tuple[float, float, float],
                        neighbor_info: NodeInfo):
        """Update neighbor information"""
        state = self.get_or_create_state(node_id)
        state.update_neighbor_info(neighbor_id, neighbor_info)
        state.record_encounter(neighbor_id, rssi, position)
    
    def record_encounter(self, node_id: str, neighbor_id: str,
                        rssi: float, position: Tuple[float, float, float],
                        forwarded_packets: List[str]):
        """Record an encounter between nodes"""
        state = self.get_or_create_state(node_id)
        state.record_encounter(neighbor_id, rssi, position, forwarded_packets)
    
    def update_packet_stats(self, node_id: str, action: str):
        """Update packet statistics for a node"""
        state = self.get_or_create_state(node_id)
        state.update_packet_stats(action)
    
    def merge(self, other: 'GCounter'):
        """
        Merge with another counter using timestamp-based conflict resolution
        Ensures:
        - Commutativity: merge(a,b) = merge(b,a)
        - Associativity: merge(merge(a,b),c) = merge(a,merge(b,c))
        - Idempotence: merge(a,a) = a
        """
        if not isinstance(other, GCounter):
            error("Cannot merge with non-GCounter object")
            return
            
        all_nodes = set(self.states.keys()) | set(other.states.keys())
        
        for node_id in all_nodes:
            self_state = self.states.get(node_id)
            other_state = other.states.get(node_id)
            
            if not self_state:
                if other_state:
                    self.states[node_id] = other_state
                continue
            
            if not other_state:
                continue
            
            # Merge based on timestamp
            if other_state.timestamp > self_state.timestamp:
                # Update node info if newer
                if other_state.node_info:
                    self_state.node_info = other_state.node_info
                
                # Merge neighbors (keep newer information)
                for n_id, n_info in other_state.neighbors.items():
                    if (n_id not in self_state.neighbors or
                        n_info.last_update > self_state.neighbors[n_id].last_update):
                        self_state.neighbors[n_id] = n_info
                
                # Merge encounter history (keep most recent encounter per neighbor)
                for n_id, encounter in other_state.encounter_history.items():
                    if (n_id not in self_state.encounter_history or
                        encounter['timestamp'] > self_state.encounter_history[n_id]['timestamp']):
                        self_state.encounter_history[n_id] = encounter
                
                # Merge packet stats (take max values)
                for stat in self_state.packet_stats:
                    self_state.packet_stats[stat] = max(
                        self_state.packet_stats[stat],
                        other_state.packet_stats.get(stat, 0)
                    )
                
                # Update link quality and congestion level
                self_state.link_quality.update(other_state.link_quality)
                self_state.congestion_level = max(
                    self_state.congestion_level,
                    other_state.congestion_level
                )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            node_id: state.to_dict()
            for node_id, state in self.states.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GCounter':
        """Create GCounter from dictionary"""
        counter = cls()
        if not isinstance(data, dict):
            error("Invalid data format for GCounter.from_dict")
            return counter
            
        try:
            for node_id, state_data in data.items():
                counter.states[node_id] = NetworkState.from_dict(state_data)
        except Exception as e:
            error(f"Error creating GCounter from dict: {str(e)}")
            import traceback
            error(traceback.format_exc())
        return counter

class OrSet:
    """
    Observed-Remove Set CRDT supporting conflict-free add/remove operations
    Each element is tagged with a unique identifier ⟨packet_id, tag⟩, 
    where tag is a unique identifier ⟨sender_id:timestamp⟩
    """
    def __init__(self):
        self._pending = {}  # packet_id -> {(sender_id, timestamp)}
        self._acknowledged = {}  # packet_id -> {(sender_id, timestamp)}
        self.MAX_ELEMENTS = 100  # Maximum number of elements to store per packet_id
    
    def add(self, packet_id, sender_id=None, timestamp=None):
        """
        Add a packet_id to the set with the given tag
        If sender_id or timestamp not provided, generate defaults
        """
        if sender_id is None:
            sender_id = "default"
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize set for this packet_id if needed
        if packet_id not in self._pending:
            self._pending[packet_id] = set()
        
        # Add the tag
        self._pending[packet_id].add((sender_id, timestamp))
        
        # Limit size if needed
        if len(self._pending[packet_id]) > self.MAX_ELEMENTS:
            # Convert to list and sort by timestamp to keep most recent
            sorted_tags = sorted(self._pending[packet_id], key=lambda x: x[1])
            self._pending[packet_id] = set(sorted_tags[-self.MAX_ELEMENTS:])
    
    def remove(self, packet_id, sender_id=None, timestamp=None):
        """
        Remove a packet_id from the set by moving its tags to removed set
        """
        if sender_id is None:
            sender_id = "default"
        if timestamp is None:
            timestamp = time.time()
            
        # Check if the packet exists in the added set
        if packet_id in self._pending and self._pending[packet_id]:
            # Initialize removed set for this packet if needed
            if packet_id not in self._acknowledged:
                self._acknowledged[packet_id] = set()
            
            # Move tags from added to removed
            self._acknowledged[packet_id].update(self._pending[packet_id])
            
            # Clear the added set for this packet
            self._pending[packet_id] = set()
            
            # Limit size of removed set if needed
            if len(self._acknowledged[packet_id]) > self.MAX_ELEMENTS:
                sorted_tags = sorted(self._acknowledged[packet_id], key=lambda x: x[1])
                self._acknowledged[packet_id] = set(sorted_tags[-self.MAX_ELEMENTS:])
    
    def merge(self, other):
        """
        Merge with another OR-Set
        Implementation of observed-remove set semantics:
        1. All added elements from other are merged into this set
        2. All removed elements from other are merged into this set
        3. Elements are considered "in the set" if they have adds not covered by removes
        """
        if not isinstance(other, OrSet):
            error("Cannot merge with non-OrSet object")
            return
        
        # Merge added elements
        for packet_id, tags in other._pending.items():
            if packet_id not in self._pending:
                self._pending[packet_id] = set()
            self._pending[packet_id].update(tags)
            
            # Apply size limit
            if len(self._pending[packet_id]) > self.MAX_ELEMENTS:
                sorted_tags = sorted(self._pending[packet_id], key=lambda x: x[1])
                self._pending[packet_id] = set(sorted_tags[-self.MAX_ELEMENTS:])
        
        # Merge removed elements
        for packet_id, tags in other._acknowledged.items():
            if packet_id not in self._acknowledged:
                self._acknowledged[packet_id] = set()
            self._acknowledged[packet_id].update(tags)
            
            # Apply size limit
            if len(self._acknowledged[packet_id]) > self.MAX_ELEMENTS:
                sorted_tags = sorted(self._acknowledged[packet_id], key=lambda x: x[1])
                self._acknowledged[packet_id] = set(sorted_tags[-self.MAX_ELEMENTS:])
            
            # Remove tags that are in the removed set from the added set
            if packet_id in self._pending:
                self._pending[packet_id] -= self._acknowledged[packet_id]
    
    @property
    def elements(self):
        """
        Get the effective elements in the set (added but not removed)
        Returns a list of (packet_id, (sender_id, timestamp)) tuples
        """
        result = []
        for packet_id, tags in self._pending.items():
            if tags:  # Only include if there are tags not removed
                for tag in tags:
                    result.append((packet_id, tag))
        
        # Apply global size limit if needed
        if len(result) > self.MAX_ELEMENTS:
            # Sort by timestamp
            result.sort(key=lambda x: x[1][1])
            result = result[-self.MAX_ELEMENTS:]
        
        return result
    
    @property 
    def pending_packets(self):
        """Get packet IDs that have been added but not acknowledged"""
        # Create a copy of the items to avoid dictionary modification during iteration
        items = list(self._pending.items())
        return [packet_id for packet_id, tags in items if tags]

    @property
    def acknowledged_packets(self):
        """Get packet IDs that have been acknowledged (removed)"""
        return list(self._acknowledged.keys())
    
    @elements.setter
    def elements(self, element_list):
        """
        Set elements from a deserialized list
        Each element should be a tuple (packet_id, (sender_id, timestamp))
        """
        self._pending = {}
        self._acknowledged = {}
        
        for item in element_list:
            if isinstance(item, tuple) and len(item) == 2:
                packet_id, tag = item
                if isinstance(tag, tuple) and len(tag) == 2:
                    sender_id, timestamp = tag
                    self.add(packet_id, sender_id, timestamp)
                else:
                    # Backward compatibility for simple packet IDs
                    self.add(packet_id)
            else:
                # Backward compatibility for simple packet IDs
                self.add(item)
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'added': {pid: list(tags) for pid, tags in self._pending.items()},
            'removed': {pid: list(tags) for pid, tags in self._acknowledged.items()}
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary representation"""
        orset = cls()
        
        if 'added' in data:
            for pid, tags in data['added'].items():
                orset._pending[pid] = set(tuple(tag) for tag in tags)
        
        if 'removed' in data:
            for pid, tags in data['removed'].items():
                orset._acknowledged[pid] = set(tuple(tag) for tag in tags)
        
        return orset 