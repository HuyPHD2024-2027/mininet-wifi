import time
import json
import math
from collections import defaultdict, deque

class NodeStateTracker:
    """Tracks state information about nodes for reinforcement learning"""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.position_history = deque(maxlen=max_history)
        self.rssi_history = defaultdict(lambda: deque(maxlen=max_history))
        self.encounter_history = defaultdict(lambda: deque(maxlen=max_history))
        self.neighbor_data = {}
        self.last_position = None
        self.battery_level = 100  # Simulated battery level (%)
        self.storage_used = 0     # Simulated storage used (KB)
        self.storage_capacity = 1000  # Simulated storage capacity (KB)
        self.mobility_pattern = "unknown"  # Mobility pattern classification
        self.network_congestion = 0.0  # Network congestion level (0-1)
        self.last_update_time = time.time()
    
    def update_position(self, position):
        """Update node position and track history"""
        timestamp = time.time()
        if self.last_position:
            # Calculate speed based on position change
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            time_diff = timestamp - self.last_update_time
            speed = distance / time_diff if time_diff > 0 else 0
            
            self.position_history.append({
                'position': position,
                'timestamp': timestamp,
                'speed': speed,
                'direction': math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0
            })
            
            # Update mobility pattern if we have enough history
            if len(self.position_history) > 10:
                self._update_mobility_pattern()
        else:
            self.position_history.append({
                'position': position,
                'timestamp': timestamp,
                'speed': 0,
                'direction': 0
            })
        
        self.last_position = position
        self.last_update_time = timestamp
    
    def _update_mobility_pattern(self):
        """Analyze position history to determine mobility pattern"""
        # Simple mobility pattern classification based on recent movement
        speeds = [entry['speed'] for entry in list(self.position_history)[-10:]]
        directions = [entry['direction'] for entry in list(self.position_history)[-10:]]
        
        avg_speed = sum(speeds) / len(speeds)
        
        # Calculate direction variance
        dir_variance = 0
        if len(directions) > 1:
            mean_dir = sum(directions) / len(directions)
            dir_variance = sum((d - mean_dir)**2 for d in directions) / len(directions)
        
        if avg_speed < 0.1:
            self.mobility_pattern = "stationary"
        elif dir_variance < 0.2:
            self.mobility_pattern = "linear"
        elif 0.2 <= dir_variance < 1.0:
            self.mobility_pattern = "random_waypoint"
        else:
            self.mobility_pattern = "random_walk"
    
    def update_rssi(self, neighbor_id, rssi):
        """Update RSSI history for a neighbor"""
        self.rssi_history[neighbor_id].append({
            'rssi': rssi,
            'timestamp': time.time()
        })
    
    def record_encounter(self, neighbor_id, rssi, neighbor_position):
        """Record an encounter with a neighbor"""
        timestamp = time.time()
        
        # Update RSSI history
        self.update_rssi(neighbor_id, rssi)
        
        # Record encounter details
        self.encounter_history[neighbor_id].append({
            'timestamp': timestamp,
            'rssi': rssi,
            'position': self.last_position,
            'neighbor_position': neighbor_position
        })
        
        # Update neighbor data
        if neighbor_id not in self.neighbor_data:
            self.neighbor_data[neighbor_id] = {
                'first_encounter': timestamp,
                'encounter_count': 0,
                'successful_transfers': 0,
                'failed_transfers': 0,
                'avg_rssi': rssi,
                'position': neighbor_position,
                'last_encounter': timestamp,
            }
        else:
            # Update existing neighbor data
            neighbor = self.neighbor_data[neighbor_id]
            rssi_values = [e['rssi'] for e in self.rssi_history[neighbor_id]]
            
            neighbor['encounter_count'] += 1
            neighbor['last_encounter'] = timestamp
            neighbor['position'] = neighbor_position
            neighbor['avg_rssi'] = sum(rssi_values) / len(rssi_values)
            
            # Calculate encounter frequency (encounters per hour)
            time_diff_hours = (timestamp - neighbor['first_encounter']) / 3600
            if time_diff_hours > 0:
                neighbor['encounter_frequency'] = neighbor['encounter_count'] / time_diff_hours
            else:
                neighbor['encounter_frequency'] = 0
        
        # Simulate battery usage for encounter
        self.battery_level -= 0.01  # Small battery drain for encounter
        if self.battery_level < 0:
            self.battery_level = 0
    
    def record_packet_transfer(self, neighbor_id, success):
        """Record the success or failure of a packet transfer"""
        if neighbor_id in self.neighbor_data:
            if success:
                self.neighbor_data[neighbor_id]['successful_transfers'] += 1
            else:
                self.neighbor_data[neighbor_id]['failed_transfers'] += 1
            
            # Update success rate
            total = (self.neighbor_data[neighbor_id]['successful_transfers'] + 
                    self.neighbor_data[neighbor_id]['failed_transfers'])
            
            if total > 0:
                self.neighbor_data[neighbor_id]['success_rate'] = (
                    self.neighbor_data[neighbor_id]['successful_transfers'] / total
                )
    
    def get_node_features(self):
        """Get features for the node useful for RL"""
        # Calculate metrics based on history
        current_time = time.time()
        encounter_frequency = {}
        for neighbor_id, encounters in self.encounter_history.items():
            if len(encounters) > 1:
                # Calculate average time between encounters
                times = [e['timestamp'] for e in encounters]
                intervals = [times[i] - times[i-1] for i in range(1, len(times))]
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    encounter_frequency[neighbor_id] = 3600 / avg_interval if avg_interval > 0 else 0  # Per hour
        
        # Create feature vector
        features = {
            'position': self.last_position,
            'battery_level': self.battery_level,
            'storage_available': (self.storage_capacity - self.storage_used) / self.storage_capacity,
            'neighbor_count': len(self.neighbor_data),
            'active_neighbors': sum(1 for n_id, n in self.neighbor_data.items() 
                                 if current_time - n['last_encounter'] < 300),  # Active in last 5 min
            'mobility_pattern': self.mobility_pattern,
            'network_congestion': self.network_congestion,
            'avg_encounter_frequency': sum(encounter_frequency.values()) / len(encounter_frequency) 
                                      if encounter_frequency else 0,
            'neighbors': {n_id: {
                'success_rate': n.get('success_rate', 0),
                'avg_rssi': n.get('avg_rssi', -100),
                'encounter_frequency': n.get('encounter_frequency', 0),
                'last_encounter_age': current_time - n['last_encounter'],
                'position': n.get('position')
            } for n_id, n in self.neighbor_data.items()}
        }
        
        return features
    
    def to_dict(self):
        """Convert tracker state to dictionary for serialization"""
        return {
            'position': self.last_position,
            'battery_level': self.battery_level,
            'storage_used': self.storage_used,
            'storage_capacity': self.storage_capacity,
            'mobility_pattern': self.mobility_pattern,
            'network_congestion': self.network_congestion,
            'neighbor_data': self.neighbor_data,
            'last_update_time': self.last_update_time
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create tracker from dictionary"""
        tracker = cls()
        tracker.last_position = data.get('position')
        tracker.battery_level = data.get('battery_level', 100)
        tracker.storage_used = data.get('storage_used', 0)
        tracker.storage_capacity = data.get('storage_capacity', 1000)
        tracker.mobility_pattern = data.get('mobility_pattern', 'unknown')
        tracker.network_congestion = data.get('network_congestion', 0.0)
        tracker.neighbor_data = data.get('neighbor_data', {})
        tracker.last_update_time = data.get('last_update_time', time.time())
        return tracker


class PacketTracker:
    """Tracks packet forwarding history for reinforcement learning"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.received_packets = {}  # packet_id -> packet_data
        self.successful_deliveries = {}  # packet_id -> delivery_data
        self.failed_deliveries = {}  # packet_id -> failure_data
        self.packet_history = deque(maxlen=max_history)
    
    def record_forwarded_packet(self, packet_id, packet_data):
        """Record a packet being forwarded"""
        timestamp = time.time()
        
        packet_data = {
            'packet_id': packet_id,
            'source': packet_data.get('source'),
            'destination': packet_data.get('destination'),
            'size': packet_data.get('size', 0),
            'priority': packet_data.get('priority', 1),
            'creation_time': packet_data.get('timestamp', timestamp),
            'forwarded_time': timestamp,
            'hops': packet_data.get('hops', 0),
            'ttl': packet_data.get('ttl', 64)
        }
        
        self.received_packets[packet_id] = packet_data
        
        # Add to history
        self.packet_history.append({
            'type': 'forward',
            'timestamp': timestamp,
            'packet_id': packet_id,
            'packet_data': packet_data
        })
    
    def record_delivery(self, packet_id, success, delay=None, reason=None):
        """Record packet delivery success or failure"""
        timestamp = time.time()
        
        if packet_id in self.received_packets:
            packet_data = self.received_packets[packet_id]
            
            if success:
                delivery_data = {
                    'packet_id': packet_id,
                    'delivered_time': timestamp,
                    'delay': delay or (timestamp - packet_data['forwarded_time']),
                    'hops': packet_data['hops']
                }
                self.successful_deliveries[packet_id] = delivery_data
            else:
                failure_data = {
                    'packet_id': packet_id,
                    'failed_time': timestamp,
                    'reason': reason or 'unknown',
                    'age': timestamp - packet_data['forwarded_time']
                }
                self.failed_deliveries[packet_id] = failure_data
            
            # Add to history
            self.packet_history.append({
                'type': 'delivery' if success else 'failure',
                'timestamp': timestamp,
                'packet_id': packet_id,
                'data': delivery_data if success else failure_data
            })
    
    def get_packet_statistics(self):
        """Get statistics about packet forwarding and delivery"""
        successful_count = len(self.successful_deliveries)
        failed_count = len(self.failed_deliveries)
        total_forwarded = len(self.received_packets)
        
        if total_forwarded > 0:
            success_rate = successful_count / total_forwarded
        else:
            success_rate = 0
            
        # Calculate average delivery delay for successful deliveries
        delays = [data['delay'] for data in self.successful_deliveries.values()]
        avg_delay = sum(delays) / len(delays) if delays else 0
        
        # Calculate hop counts
        hop_counts = [data['hops'] for data in self.successful_deliveries.values()]
        avg_hops = sum(hop_counts) / len(hop_counts) if hop_counts else 0
        
        return {
            'total_forwarded': total_forwarded,
            'successful_deliveries': successful_count,
            'failed_deliveries': failed_count,
            'success_rate': success_rate,
            'average_delay': avg_delay,
            'average_hops': avg_hops,
            'packet_ids': list(self.received_packets.keys())
        }
    
    def to_dict(self):
        """Convert tracker state to dictionary for serialization"""
        return {
            'received_packets': self.received_packets,
            'successful_deliveries': self.successful_deliveries,
            'failed_deliveries': self.failed_deliveries
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create tracker from dictionary"""
        tracker = cls()
        tracker.received_packets = data.get('received_packets', {})
        tracker.successful_deliveries = data.get('successful_deliveries', {})
        tracker.failed_deliveries = data.get('failed_deliveries', {})
        return tracker 