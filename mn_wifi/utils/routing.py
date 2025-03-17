from mininet.log import info
import time

def calculate_neighbor_metrics(node_state, neighbor_name, destination, dest_position=None):
    """Calculate neighbor metrics for forwarding decision"""
    metrics_score = 0.5  # Default score
    
    # Check if we have state for this neighbor
    if neighbor_name in node_state.states:
        state = node_state.states[neighbor_name]
        
        # 1. Check link quality to this neighbor
        if neighbor_name in state.link_quality:
            link_quality = state.link_quality[neighbor_name]
            metrics_score += link_quality * 0.2  # Weight link quality by 0.2
        
        # 2. Check if neighbor has encountered destination
        if destination in state.encounter_history:
            encounter = state.encounter_history[destination]
            
            # More recent encounters are weighted higher
            recency = 1.0 - min(1.0, (time.time() - encounter['timestamp']) / (60 * 60))  # Scale by hour
            metrics_score += recency * 0.3  # Weight recency by 0.3
            
            # If we know destination position, calculate distance
            if dest_position and 'position' in encounter:
                from mn_wifi.utils.network_utils import calculate_distance
                distance = calculate_distance(dest_position, encounter['position'])
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

def is_good_next_hop(node, neighbor, destination, network_state):
    """Determine if a neighbor is a good next hop for a destination"""
    # If crdt data is not available, just return True as default
    if not network_state:
        return True
        
    # Get destination node's last known position
    from mn_wifi.constants import get_node_by_name
    dest_position = None
    dest_node = get_node_by_name(destination)
    if dest_node and hasattr(dest_node, 'position'):
        dest_position = dest_node.position
    
    # If destination node is unknown or position not available,
    # check if neighbor has encountered destination before
    for node_id, state in network_state.states.items():
        if destination in state.encounter_history:
            # Neighbor has encountered destination before
            return True
            
    # Calculate metrics for forwarding decision
    neighbor_metrics = calculate_neighbor_metrics(network_state, neighbor.name, destination, dest_position)
    
    # Simple decision: if metrics score > 0.5, consider it a good next hop
    info(f"*** {node.name} neighbor {neighbor.name} metrics: {neighbor_metrics}\n")
    return neighbor_metrics > 0.2 