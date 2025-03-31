from mininet.log import info, error
import time

class Colors:
    # ANSI escape codes for colors
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    ORANGE = '\033[93m'
    RESET = '\033[0m'

class OpportunisticLogger:
    # Define color scheme for different event types
    EVENT_COLORS = {
        'Packet Delivery': Colors.GREEN,
        'Packet Sent': Colors.GREEN,
        'CRDT Sync': Colors.BLUE,
        'Packet Storage': Colors.YELLOW,
        'Retransmission Added': Colors.GREEN,
        'Packet Generation': Colors.YELLOW,
        'No Known Neighbors': Colors.YELLOW,
        'Bundle Check Summary': Colors.BLUE,
        'CRDT Update': Colors.BLUE,
        'Invalid Payload': Colors.RED,
        'Expired Bundle': Colors.RED,
        'Packet Generation Error': Colors.RED,
        'No Destinations for Packet Generation': Colors.RED,
        'CRDT Error': Colors.RED,
        'Error': Colors.RED,
        'Packet Generation Error': Colors.RED,
        'CRDT Update Error': Colors.RED,
        'CRDT Packets Full': Colors.ORANGE,
        'Packet Stored': Colors.YELLOW,
        'Packet Forwarded': Colors.GREEN,
    }

    @staticmethod
    def log_event(event_type, details, is_error=False):
        """
        Centralized logging function for opportunistic network events with color coding
        Args:
            event_type (str): Type of event (e.g., 'CRDT Sync', 'Packet Generation')
            details (dict): Dictionary containing event details
            is_error (bool): Whether this is an error log
        """
        timestamp = time.strftime("%H:%M:%S")
        
        # Get color for event type, default to RESET if not specified
        color = OpportunisticLogger.EVENT_COLORS.get(event_type, Colors.RESET)
        
        # Create colored message
        log_message = f"\n{color}*** {event_type} [{timestamp}]{Colors.RESET}\n"
        
        # Add details with consistent color
        for key, value in details.items():
            log_message += f"{color}    - {key}: {value}{Colors.RESET}\n"
            
        if is_error:
            error(log_message)
        else:
            info(log_message)

    @staticmethod
    def add_event_color(event_type, color):
        """
        Add or update color for an event type
        Args:
            event_type (str): Type of event
            color (str): ANSI color code
        """
        OpportunisticLogger.EVENT_COLORS[event_type] = color

# Create singleton instance
logger = OpportunisticLogger()

# Example usage:
# logger.log_event("Packet Delivery", {"From": "node1", "To": "node2"})  # Will be green
# logger.log_event("CRDT Sync", {"Status": "Success"})  # Will be blue
# logger.log_event("Error", {"Message": "Failed"}, is_error=True)  # Will be red

# logger.add_event_color("New Event Type", Colors.BLUE) 