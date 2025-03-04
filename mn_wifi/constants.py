"""Constants for Mininet-WiFi"""

from mininet.log import info, error

# Global dictionaries for node discovery
STATION_MAC_MAPPING = {}  # Maps MAC addresses to station names
NODE_REGISTRY = {}        # Maps node names to node objects

def register_station_mac(station_name, mac):
    """Register a station name to MAC address mapping"""
    STATION_MAC_MAPPING[mac] = station_name
    # info(f"Registered MAC: {station_name} -> {mac}\n")

def register_node(node_name, node_obj):
    """Register a node in the global registry"""
    NODE_REGISTRY[node_name] = node_obj
    info(f"Registered node: {node_name}\n")

def get_node_by_name(node_name):
    """Get node object by name"""
    return NODE_REGISTRY.get(node_name)

def get_node_by_mac(mac):
    """Get node object by MAC address"""
    station_name = STATION_MAC_MAPPING.get(mac)
    if station_name:
        return NODE_REGISTRY.get(station_name)
    return None

# Reverse mapping for looking up MACs by station name
STATION_NAME_MAPPING = {v: k for k, v in STATION_MAC_MAPPING.items()} 