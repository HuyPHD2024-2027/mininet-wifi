import time
import numpy as np
import matplotlib.pyplot as plt
from mininet.log import info
from mn_wifi.crdt import GCounter, OrSet

class ExperimentMetrics:
    def __init__(self):
        self.delivery_rates = []
        self.end_to_end_delays = []
        self.resource_usage = []
        self.crdt_overhead = []
        self.convergence_times = []
        self.contact_frequencies = []
        self.rl_rewards = []
        
    def plot_all_metrics(self, save_dir='results/'):
        """Plot all experimental metrics"""
        # Create results directory if it doesn't exist
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot delivery rates
        self._plot_delivery_rates(save_dir)
        self._plot_delays(save_dir)
        self._plot_resource_usage(save_dir)
        self._plot_crdt_overhead(save_dir)
        self._plot_learning_curve(save_dir)
        
    def _plot_delivery_rates(self, save_dir):
        plt.figure(figsize=(10, 6))
        areas = [f"{a[0]}x{a[1]}" for a, _ in self.delivery_rates]
        rates = [r for _, r in self.delivery_rates]
        plt.bar(areas, rates)
        plt.title('Message Delivery Rate vs Network Density')
        plt.xlabel('Area Size')
        plt.ylabel('Delivery Rate')
        plt.savefig(f'{save_dir}/delivery_rates.png')
        plt.close()
        
    def _plot_delays(self, save_dir):
        plt.figure(figsize=(10, 6))
        plt.hist(self.end_to_end_delays, bins=20)
        plt.title('End-to-End Delay Distribution')
        plt.xlabel('Delay (seconds)')
        plt.ylabel('Frequency')
        plt.savefig(f'{save_dir}/delays.png')
        plt.close()
        
    def _plot_resource_usage(self, save_dir):
        plt.figure(figsize=(10, 6))
        times = [t for t, _ in self.resource_usage]
        memory = [m for _, m in self.resource_usage]
        plt.plot(times, memory)
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory (MB)')
        plt.savefig(f'{save_dir}/resource_usage.png')
        plt.close()
        
    def _plot_crdt_overhead(self, save_dir):
        plt.figure(figsize=(10, 6))
        plt.plot(self.crdt_overhead)
        plt.title('CRDT State Size Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('State Size (bytes)')
        plt.savefig(f'{save_dir}/crdt_overhead.png')
        plt.close()
        
    def _plot_learning_curve(self, save_dir):
        plt.figure(figsize=(10, 6))
        plt.plot(self.rl_rewards)
        plt.title('RL Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.savefig(f'{save_dir}/learning_curve.png')
        plt.close()

def run_delivery_experiment(net, duration=300, area_sizes=[(30,30), (50,50), (70,70)]):
    """Experiment 1: Message Delivery vs Network Density"""
    metrics = ExperimentMetrics()
    sta1, sta2, sta3 = net.get('sta1'), net.get('sta2'), net.get('sta3')
    
    for area in area_sizes:
        info(f"\n*** Testing area size {area} ***\n")
        net.setMobilityModel(max_x=area[0], max_y=area[1])
        net.plotGraph(max_x=area[0], max_y=area[1])
    
        messages = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if int((time.time() - start_time) % 30) == 0:
                msg_id = f'msg_{len(messages)}'
                sta1.store_bundle(msg_id, f"Test message {len(messages)}", ttl=120)
                messages.append({
                    'id': msg_id,
                    'sent_time': time.time(),
                    'delivered': False
                })
                
            # Check deliveries and record delays
            for msg in messages:
                if not msg['delivered']:
                    for node in [sta2, sta3]:
                        if msg['id'] in node.bundle_store:
                            msg['delivered'] = True
                            delay = time.time() - msg['sent_time']
                            metrics.end_to_end_delays.append(delay)
            
            # Record resource usage
            memory_usage = sum(len(node.bundle_store) for node in [sta1, sta2, sta3])
            metrics.resource_usage.append((time.time() - start_time, memory_usage))
            
            # Record CRDT overhead
            crdt_size = sum(len(str(node.rl_states_counter.counters)) + 
                          len(str(node.forwarded_packets.elements))
                          for node in [sta1, sta2, sta3])
            metrics.crdt_overhead.append(crdt_size)
            
            time.sleep(1)
            
        delivery_rate = len([m for m in messages if m['delivered']]) / len(messages)
        metrics.delivery_rates.append((area, delivery_rate))
        
    return metrics

def run_convergence_experiment(net, trials=5):
    """Experiment 2: CRDT Convergence Analysis"""
    metrics = ExperimentMetrics()
    sta1, sta2, sta3 = net.get('sta1'), net.get('sta2'), net.get('sta3')
    
    for trial in range(trials):
        # Reset CRDT states
        for node in [sta1, sta2, sta3]:
            node.rl_states_counter = GCounter()
            node.forwarded_packets = OrSet()
            
        start_time = time.time()
        sta1.store_bundle('conv_test', "Convergence test", ttl=120)
        
        while True:
            states = [node.rl_states_counter.value() for node in [sta1, sta2, sta3]]
            if len(set(states)) == 1:
                metrics.convergence_times.append(time.time() - start_time)
                break
            time.sleep(1)
            
    return metrics

def run_resource_experiment(net, duration=300):
    """Experiment 3: Resource Usage Analysis"""
    metrics = ExperimentMetrics()
    sta1, sta2, sta3 = net.get('sta1'), net.get('sta2'), net.get('sta3')
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # Generate periodic messages
        if int((time.time() - start_time) % 10) == 0:
            sta1.store_bundle(f'res_msg_{time.time()}', "Resource test", ttl=120)
            
        # Record memory usage
        memory = sum(len(node.bundle_store) for node in [sta1, sta2, sta3])
        metrics.resource_usage.append((time.time() - start_time, memory))
        
        # Record CRDT overhead
        crdt_size = sum(len(str(node.rl_states_counter.counters)) + 
                       len(str(node.forwarded_packets.elements))
                       for node in [sta1, sta2, sta3])
        metrics.crdt_overhead.append(crdt_size)
        
        time.sleep(1)
        
    return metrics 