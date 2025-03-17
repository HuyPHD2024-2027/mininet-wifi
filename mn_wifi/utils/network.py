import socket
import random
from threading import Lock
from mininet.log import error, info
import json
from queue import Queue
import time

class PortManager:
    """Utility class for managing ports"""
    _used_ports = set()
    _port_lock = Lock()
    
    @classmethod
    def get_available_port(cls, start_port=9000, end_port=9999):
        """Get an available port number"""
        with cls._port_lock:
            available_ports = set(range(start_port, end_port + 1)) - cls._used_ports
            if not available_ports:
                raise RuntimeError("No available ports")
            port = random.choice(list(available_ports))
            cls._used_ports.add(port)
            return port

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5

def send_packet(packet, dest_port, host='127.0.0.1'):
    """Send a packet to a destination"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Convert packet to dictionary or use packet as is if it's already a dict
        if hasattr(packet, 'to_dict'):
            packet_data = packet.to_dict()
        else:
            packet_data = packet
            
        packet_json = json.dumps(packet_data)
        sock.sendto(packet_json.encode(), (host, dest_port))
        return True
    except Exception as e:
        error(f"Error sending packet: {str(e)}\n")
        return False
    finally:
        sock.close() 