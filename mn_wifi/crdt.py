class GCounter:
    """Grow-only Counter CRDT"""
    def __init__(self):
        self.counters = {}
    
    def increment(self, node_id, value=1):
        """Increment counter for a node"""
        self.counters[node_id] = self.counters.get(node_id, 0) + value
    
    def value(self):
        """Get total counter value"""
        return sum(self.counters.values())
    
    def merge(self, other):
        """Merge with another counter"""
        for node_id, count in other.counters.items():
            self.counters[node_id] = max(self.counters.get(node_id, 0), count)

class OrSet:
    """Observed-Remove Set CRDT"""
    def __init__(self):
        self.elements = {}  # {element -> {tag -> bool}}
    
    def add(self, element, tag):
        """Add element with a tag"""
        if element not in self.elements:
            self.elements[element] = {}
        self.elements[element][tag] = True
    
    def remove(self, element):
        """Remove all tags for an element"""
        if element in self.elements:
            del self.elements[element]
    
    def contains(self, element):
        """Check if element exists in the set"""
        return element in self.elements and bool(self.elements[element])
    
    def merge(self, other):
        """Merge with another set"""
        for element, tags in other.elements.items():
            if element not in self.elements:
                self.elements[element] = {}
            self.elements[element].update(tags) 