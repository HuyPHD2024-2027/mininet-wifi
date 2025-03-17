"""Bundle store management utilities"""
from queue import Queue
from mininet.log import info
import time

class BundleManager:
    def __init__(self):
        self.bundle_store = Queue()
        
    def store_bundle(self, packet, ttl=300):
        """Store a packet in the bundle store"""
        self.bundle_store.put({
            'packet': packet,
            'stored_at': time.time(),
            'ttl': ttl
        })
        info(f"*** Stored packet {packet.packet_id} in bundle store\n")
        return packet.packet_id
        
    def process_bundles(self, neighbor, network_state, max_forwards=5):
        """Process all bundles and attempt forwarding"""
        current_time = time.time()
        temp_queue = Queue()
        forwarded_count = 0
        
        while not self.bundle_store.empty():
            bundle = self.bundle_store.get()
            
            if self._should_forward_bundle(bundle, current_time, neighbor, 
                                         network_state, forwarded_count, max_forwards):
                if self._attempt_forward(bundle['packet'], neighbor):
                    forwarded_count += 1
                    continue
                    
            temp_queue.put(bundle)
            
        self.bundle_store = temp_queue
        return forwarded_count
        
    def _should_forward_bundle(self, bundle, current_time, neighbor, 
                             network_state, forwarded_count, max_forwards):
        """Determine if bundle should be forwarded"""
        # Check expiry
        if current_time - bundle['stored_at'] > bundle['ttl']:
            return False
            
        # Check forwarding limit
        if forwarded_count >= max_forwards:
            return False
            
        # Use network state for routing decision
        from mn_wifi.utils.routing_utils import is_good_next_hop
        return is_good_next_hop(self, neighbor, bundle['packet']['destination'], network_state) 