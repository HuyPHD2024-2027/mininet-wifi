import numpy as np
import time

class ForwardingAgent:
    """RL agent for making forwarding decisions"""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        
    def get_state(self, current_node, destination, contact_history):
        """Get state representation including global knowledge"""
        # Basic contact-based features
        contacts = len([c for c in contact_history if c['node'] == destination])
        last_contact = min([time.time() - c['timestamp'] 
                          for c in contact_history 
                          if c['node'] == destination], 
                         default=float('inf'))
        
        # Include global state from G-Counter
        global_state = current_node.network_state.value()
        
        return {
            'contacts': contacts,
            'last_contact': last_contact,
            'global_state': global_state
        }
        
    def get_next_hop(self, current_node, destination, contact_history):
        """Make forwarding decision using Q-learning with global state"""
        state = self.get_state(current_node, destination, contact_history)
        state_key = (state['contacts'], state['last_contact'], state['global_state'])
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Use epsilon-greedy with global state consideration
        if np.random.random() < self.epsilon:
            return self.random_neighbor(current_node)
        
        if self.q_table[state_key]:
            return max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
        return None
        
    def update(self, state, action, reward, next_state):
        """Update Q-values based on reward"""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
            
        old_value = self.q_table[state][action]
        next_max = max(self.q_table.get(next_state, {}).values(), default=0)
        
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value 