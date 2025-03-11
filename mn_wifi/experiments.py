import time
import numpy as np
import matplotlib.pyplot as plt
from mininet.log import info
from mn_wifi.crdt import GCounter, OrSet
import json
import os
import pandas as pd
from multiprocessing import Process, Queue
import psutil

class PerformanceMetrics:
    """Tracks comprehensive performance metrics for experiments"""
    
    def __init__(self):
        # Basic metrics
        self.delivery_rates = []
        self.end_to_end_delays = []
        self.resource_usage = []
        self.crdt_overhead = []
        self.convergence_times = []
        
        # Advanced metrics
        self.throughput = []  # packets/sec
        self.goodput = []     # useful data/sec
        self.missing_packets = []  # percentage
        self.control_overhead = []  # bytes
        self.network_load = []  # percentage
        self.battery_consumption = []  # simulated mW
        
        # Raw data for detailed analysis
        self.packet_traces = []  # detailed packet logs
        self.experiment_timestamps = {
            'start': 0,
            'end': 0
        }
        
        # Results dir
        self.results_dir = 'results/'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def start_experiment(self):
        """Mark experiment start time"""
        self.experiment_timestamps['start'] = time.time()
    
    def end_experiment(self):
        """Mark experiment end time"""
        self.experiment_timestamps['end'] = time.time()
    
    def log_packet(self, packet_info):
        """Log detailed packet information"""
        packet_info['timestamp'] = time.time()
        self.packet_traces.append(packet_info)
    
    def calculate_metrics(self):
        """Calculate derived metrics from raw data"""
        # Calculate experiment duration
        duration = self.experiment_timestamps['end'] - self.experiment_timestamps['start']
        if duration <= 0:
            return
            
        # Analyze packet traces
        delivered = [p for p in self.packet_traces if p.get('delivered', False)]
        sent = [p for p in self.packet_traces if p.get('action') == 'sent']
        
        # Calculate throughput (all packets/sec)
        self.throughput = len(sent) / duration if sent else 0
        
        # Calculate goodput (delivered packets/sec)
        self.goodput = len(delivered) / duration if delivered else 0
        
        # Calculate missing packets percentage
        if sent:
            self.missing_packets = (len(sent) - len(delivered)) / len(sent) * 100
        
        # Calculate average end-to-end delay
        if delivered:
            delays = [p.get('delay', 0) for p in delivered]
            self.end_to_end_delays = delays
        
        # Calculate delivery rate
        if sent:
            self.delivery_rates.append(('total', len(delivered) / len(sent)))
    
    def save_to_csv(self, experiment_name):
        """Save metrics to CSV files for further analysis"""
        # Save packet traces
        df = pd.DataFrame(self.packet_traces)
        df.to_csv(f"{self.results_dir}/{experiment_name}_packet_traces.csv", index=False)
        
        # Save summary metrics
        summary = {
            'throughput': self.throughput,
            'goodput': self.goodput,
            'missing_packets': self.missing_packets,
            'avg_delay': np.mean(self.end_to_end_delays) if self.end_to_end_delays else 0,
            'delivery_rate': self.delivery_rates[-1][1] if self.delivery_rates else 0,
            'avg_overhead': np.mean(self.crdt_overhead) if self.crdt_overhead else 0,
            'experiment_duration': self.experiment_timestamps['end'] - self.experiment_timestamps['start'],
        }
        
        with open(f"{self.results_dir}/{experiment_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def plot_comparative_metrics(self, crdt_metrics, no_crdt_metrics, title_suffix=""):
        """Create comparative plots between CRDT and non-CRDT results"""
        # Create plots directory
        plots_dir = f"{self.results_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot throughput and goodput
        self._plot_bar_comparison(
            ['Throughput', 'Goodput'],
            [crdt_metrics.throughput, crdt_metrics.goodput],
            [no_crdt_metrics.throughput, no_crdt_metrics.goodput],
            'Packets/second',
            f'Throughput and Goodput {title_suffix}',
            f'{plots_dir}/throughput_goodput{title_suffix.replace(" ", "_")}.png'
        )
        
        # Plot delivery rates and missing packets
        delivery_rate_crdt = crdt_metrics.delivery_rates[-1][1] if crdt_metrics.delivery_rates else 0
        delivery_rate_no_crdt = no_crdt_metrics.delivery_rates[-1][1] if no_crdt_metrics.delivery_rates else 0
        
        self._plot_bar_comparison(
            ['Delivery Rate (%)', 'Missing Packets (%)'],
            [delivery_rate_crdt * 100, crdt_metrics.missing_packets],
            [delivery_rate_no_crdt * 100, no_crdt_metrics.missing_packets],
            'Percentage',
            f'Delivery Performance {title_suffix}',
            f'{plots_dir}/delivery_performance{title_suffix.replace(" ", "_")}.png'
        )
        
        # Plot end-to-end delay
        plt.figure(figsize=(10, 6))
        bins = np.linspace(0, max(
            max(crdt_metrics.end_to_end_delays) if crdt_metrics.end_to_end_delays else 0,
            max(no_crdt_metrics.end_to_end_delays) if no_crdt_metrics.end_to_end_delays else 0,
            1),  # ensure at least one bin if no delays
            20)
        
        plt.hist(crdt_metrics.end_to_end_delays, bins=bins, alpha=0.5, label='With CRDT')
        plt.hist(no_crdt_metrics.end_to_end_delays, bins=bins, alpha=0.5, label='Without CRDT')
        plt.xlabel('Delay (seconds)')
        plt.ylabel('Number of Packets')
        plt.title(f'End-to-End Delay Distribution {title_suffix}')
        plt.legend()
        plt.savefig(f'{plots_dir}/delay_distribution{title_suffix.replace(" ", "_")}.png')
        plt.close()
        
        # Plot overhead comparison
        if crdt_metrics.crdt_overhead and no_crdt_metrics.crdt_overhead:
            plt.figure(figsize=(10, 6))
            plt.plot(crdt_metrics.crdt_overhead, label='With CRDT')
            plt.plot(no_crdt_metrics.crdt_overhead, label='Without CRDT')
            plt.xlabel('Time (samples)')
            plt.ylabel('Overhead (bytes)')
            plt.title(f'Protocol Overhead {title_suffix}')
            plt.legend()
            plt.savefig(f'{plots_dir}/overhead{title_suffix.replace(" ", "_")}.png')
            plt.close()
        
        # Plot resource usage
        if crdt_metrics.resource_usage and no_crdt_metrics.resource_usage:
            plt.figure(figsize=(10, 6))
            crdt_times = [t for t, _ in crdt_metrics.resource_usage]
            crdt_mem = [m for _, m in crdt_metrics.resource_usage]
            no_crdt_times = [t for t, _ in no_crdt_metrics.resource_usage]
            no_crdt_mem = [m for _, m in no_crdt_metrics.resource_usage]
            
            plt.plot(crdt_times, crdt_mem, label='With CRDT')
            plt.plot(no_crdt_times, no_crdt_mem, label='Without CRDT')
            plt.xlabel('Time (s)')
            plt.ylabel('Memory Usage')
            plt.title(f'Memory Usage {title_suffix}')
            plt.legend()
            plt.savefig(f'{plots_dir}/memory_usage{title_suffix.replace(" ", "_")}.png')
            plt.close()
    
    def _plot_bar_comparison(self, labels, crdt_values, no_crdt_values, ylabel, title, filename):
        """Helper to create bar comparison plots"""
        plt.figure(figsize=(10, 6))
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, crdt_values, width, label='With CRDT')
        plt.bar(x + width/2, no_crdt_values, width, label='Without CRDT')
        
        plt.xlabel('Metrics')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x, labels)
        plt.legend()
        
        plt.savefig(filename)
        plt.close()


def run_comparative_experiment(net, duration=300, with_crdt=True, name_suffix=""):
    """Run a controlled experiment with or without CRDT"""
    metrics = PerformanceMetrics()
    metrics.start_experiment()
    
    # Get nodes
    nodes = [net.get(f'sta{i+1}') for i in range(3)]
    
    # Configure nodes based on experiment type
    if not with_crdt:
        for node in nodes:
            # Disable CRDT updates but keep packet delivery
            node.max_crdt_packets = 0
            info(f"*** Disabled CRDT updates for {node.name}\n")
    else:
        for node in nodes:
            # Ensure CRDT is enabled
            node.max_crdt_packets = 999999999
            info(f"*** Enabled CRDT updates for {node.name}\n")
    
    # Clear existing state
    for node in nodes:
        node.bundle_store = {}
        node.network_state = GCounter()
        node.received_packets = OrSet()
        node.packets_sent = 0
        node.crdt_packets_sent = 0
    
    # Generate test traffic
    source_node = nodes[0]
    total_packets = 100
    packets_sent = 0
    sent_packets = []
    
    info(f"\n*** Starting comparative experiment: {'With CRDT' if with_crdt else 'Without CRDT'} ***\n")
    
    start_time = time.time()
    packet_interval = duration / total_packets
    
    next_packet_time = start_time
    
    while time.time() - start_time < duration and packets_sent < total_packets:
        current_time = time.time()
        
        # Send packets at regular intervals
        if current_time >= next_packet_time and packets_sent < total_packets:
            # Create a test packet
            packet_id = f"test_packet_{packets_sent}"
            dest_idx = (packets_sent % (len(nodes) - 1)) + 1  # Round-robin to other nodes
            dest_node = nodes[dest_idx]
            
            info(f"*** Sending test packet {packet_id} from {source_node.name} to {dest_node.name}\n")
            
            # Store packet info for tracking
            packet_info = {
                'packet_id': packet_id,
                'source': source_node.name,
                'destination': dest_node.name,
                'sent_time': current_time,
                'action': 'sent',
                'delivered': False,
                'with_crdt': with_crdt
            }
            metrics.log_packet(packet_info)
            sent_packets.append(packet_info)
            
            # Send packet
            source_node.send_packet_to(
                dest_node.name, 
                {
                    'test_data': f"Test packet {packets_sent}",
                    'packet_id': packet_id,
                    'creation_time': current_time,
                    'size': 1024  # Simulated packet size in bytes
                }
            )
            
            packets_sent += 1
            next_packet_time = start_time + (packets_sent * packet_interval)
        
        # Check for packet deliveries
        for packet in sent_packets:
            if not packet['delivered']:
                dest_node = net.get(packet['destination'])
                
                # Check if destination received the packet
                for bundle_id, bundle in dest_node.bundle_store.items():
                    if isinstance(bundle, dict) and 'packet' in bundle:
                        bundle_packet = bundle['packet']
                        if (hasattr(bundle_packet, 'payload') and 
                            isinstance(bundle_packet.payload, dict) and
                            bundle_packet.payload.get('packet_id') == packet['packet_id']):
                            
                            # Mark as delivered
                            packet['delivered'] = True
                            packet['delivery_time'] = current_time
                            packet['delay'] = current_time - packet['sent_time']
                            
                            info(f"*** Packet {packet['packet_id']} delivered to {packet['destination']}, "
                                f"delay: {packet['delay']:.2f}s\n")
                            
                            # Update metrics
                            metrics.log_packet(packet)
                            break
        
        # Monitor resource usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        metrics.resource_usage.append((current_time - start_time, memory_usage))
        
        # Calculate CRDT overhead
        if with_crdt:
            crdt_size = sum(
                len(json.dumps(node.network_state.to_dict())) + 
                len(json.dumps(node.received_packets.elements))
                for node in nodes
            )
        else:
            crdt_size = 0
            
        metrics.crdt_overhead.append(crdt_size)
        
        # Don't busy-wait
        time.sleep(0.1)
    
    # Record final metrics
    for packet in sent_packets:
        if not packet['delivered']:
            info(f"*** Packet {packet['packet_id']} not delivered\n")
    
    # End experiment
    metrics.end_experiment()
    metrics.calculate_metrics()
    
    experiment_type = "with_crdt" if with_crdt else "no_crdt"
    if name_suffix:
        experiment_type += f"_{name_suffix}"
        
    metrics.save_to_csv(experiment_type)
    
    info(f"\n*** Experiment completed: {'With CRDT' if with_crdt else 'Without CRDT'} ***\n")
    info(f"Throughput: {metrics.throughput:.2f} packets/sec\n")
    info(f"Goodput: {metrics.goodput:.2f} packets/sec\n")
    info(f"Delivery rate: {metrics.delivery_rates[-1][1]*100:.2f}%\n")
    info(f"Missing packets: {metrics.missing_packets:.2f}%\n")
    info(f"Average delay: {np.mean(metrics.end_to_end_delays) if metrics.end_to_end_delays else 0:.2f}s\n")
    info(f"Average overhead: {np.mean(metrics.crdt_overhead) if metrics.crdt_overhead else 0:.2f} bytes\n")
    
    return metrics


def run_mobility_pattern_experiment(net, duration=300):
    """Test performance under different mobility patterns"""
    # Dictionary to store results for each pattern
    all_results = {}
    
    mobility_patterns = [
        ("random_waypoint", {"max_x": 100, "max_y": 100, "min_v": 0.5, "max_v": 2}),
        ("random_walk", {"max_x": 100, "max_y": 100, "min_v": 0.5, "max_v": 2}),
        ("reference_point", {"max_x": 100, "max_y": 100, "min_v": 0.5, "max_v": 2})
    ]
    
    for pattern_name, params in mobility_patterns:
        info(f"\n*** Testing mobility pattern: {pattern_name} ***\n")
        
        # Set mobility model
        net.setMobilityModel(
            model=pattern_name,
            **params
        )
        
        # Run with CRDT
        info(f"\n*** Running with CRDT under {pattern_name} mobility ***\n")
        crdt_metrics = run_comparative_experiment(
            net, 
            duration=duration, 
            with_crdt=True, 
            name_suffix=pattern_name
        )
        
        # Run without CRDT
        info(f"\n*** Running without CRDT under {pattern_name} mobility ***\n")
        no_crdt_metrics = run_comparative_experiment(
            net, 
            duration=duration, 
            with_crdt=False, 
            name_suffix=pattern_name
        )
        
        # Plot comparative results
        crdt_metrics.plot_comparative_metrics(
            crdt_metrics, 
            no_crdt_metrics, 
            title_suffix=f"({pattern_name})"
        )
        
        # Store results
        all_results[pattern_name] = {
            "with_crdt": crdt_metrics,
            "without_crdt": no_crdt_metrics
        }
    
    # Create summary comparison across mobility patterns
    create_mobility_comparison_summary(all_results)
    
    return all_results


def run_density_experiment(net, duration=300):
    """Test performance with different network densities"""
    # Dictionary to store results for each density
    all_results = {}
    
    # Define different network densities by adjusting the area
    density_configs = [
        ("high_density", {"max_x": 50, "max_y": 50}),
        ("medium_density", {"max_x": 100, "max_y": 100}),
        ("low_density", {"max_x": 200, "max_y": 200})
    ]
    
    for density_name, params in density_configs:
        info(f"\n*** Testing network density: {density_name} ***\n")
        
        # Configure network density
        net.setMobilityModel(
            model="random_waypoint",
            **params
        )
        
        # Run with CRDT
        info(f"\n*** Running with CRDT under {density_name} ***\n")
        crdt_metrics = run_comparative_experiment(
            net, 
            duration=duration, 
            with_crdt=True, 
            name_suffix=density_name
        )
        
        # Run without CRDT
        info(f"\n*** Running without CRDT under {density_name} ***\n")
        no_crdt_metrics = run_comparative_experiment(
            net, 
            duration=duration, 
            with_crdt=False, 
            name_suffix=density_name
        )
        
        # Plot comparative results
        crdt_metrics.plot_comparative_metrics(
            crdt_metrics, 
            no_crdt_metrics, 
            title_suffix=f"({density_name})"
        )
        
        # Store results
        all_results[density_name] = {
            "with_crdt": crdt_metrics,
            "without_crdt": no_crdt_metrics
        }
    
    # Create summary comparison across densities
    create_density_comparison_summary(all_results)
    
    return all_results


def run_network_stress_test(net, duration=300):
    """Test performance under high network load"""
    # Dictionary to store results
    all_results = {}
    
    # Define different packet rates
    packet_rates = [
        ("low_load", 5),      # 5 packets per second
        ("medium_load", 20),  # 20 packets per second
        ("high_load", 50)     # 50 packets per second
    ]
    
    for load_name, packets_per_second in packet_rates:
        info(f"\n*** Testing network load: {load_name} ({packets_per_second} packets/sec) ***\n")
        
        # Configure network
        net.setMobilityModel(model="random_waypoint", max_x=100, max_y=100)
        
        # Calculate number of packets to send
        total_packets = packets_per_second * duration
        
        # Run custom stress test
        crdt_metrics, no_crdt_metrics = run_stress_test(
            net, 
            duration=duration,
            total_packets=total_packets,
            load_name=load_name
        )
        
        # Plot comparative results
        crdt_metrics.plot_comparative_metrics(
            crdt_metrics, 
            no_crdt_metrics, 
            title_suffix=f"({load_name})"
        )
        
        # Store results
        all_results[load_name] = {
            "with_crdt": crdt_metrics,
            "without_crdt": no_crdt_metrics
        }
    
    # Create summary comparison across load conditions
    create_load_comparison_summary(all_results)
    
    return all_results


def run_stress_test(net, duration, total_packets, load_name):
    """Custom stress test with parallel packet generation"""
    # Get nodes
    nodes = [net.get(f'sta{i+1}') for i in range(3)]
    
    # First run with CRDT
    for node in nodes:
        node.max_crdt_packets = 9999999
        node.bundle_store = {}
        node.network_state = GCounter()
        node.received_packets = OrSet()
        node.packets_sent = 0
        node.crdt_packets_sent = 0
    
    info(f"\n*** Running stress test with CRDT: {load_name} ***\n")
    crdt_metrics = stress_test_worker(
        nodes, 
        duration, 
        total_packets, 
        True, 
        f"{load_name}_with_crdt"
    )
    
    # Then run without CRDT
    for node in nodes:
        node.max_crdt_packets = 0
        node.bundle_store = {}
        node.network_state = GCounter()
        node.received_packets = OrSet()
        node.packets_sent = 0
        node.crdt_packets_sent = 0
    
    info(f"\n*** Running stress test without CRDT: {load_name} ***\n")
    no_crdt_metrics = stress_test_worker(
        nodes, 
        duration, 
        total_packets, 
        False, 
        f"{load_name}_no_crdt"
    )
    
    return crdt_metrics, no_crdt_metrics


def stress_test_worker(nodes, duration, total_packets, with_crdt, name_suffix):
    """Worker function for stress tests"""
    metrics = PerformanceMetrics()
    metrics.start_experiment()
    
    # Track packet info
    sent_packets = []
    
    # Set up test parameters
    start_time = time.time()
    packet_interval = duration / total_packets
    next_packet_time = start_time
    packets_sent = 0
    
    info(f"*** Starting stress test: {name_suffix} ***\n")
    info(f"*** Target: {total_packets} packets over {duration} seconds ***\n")
    
    # Main test loop
    while time.time() - start_time < duration:
        current_time = time.time()
        
        # Send packets at regular intervals
        while current_time >= next_packet_time and packets_sent < total_packets:
            # Create and send packet
            packet_id = f"stress_{name_suffix}_{packets_sent}"
            
            # Round-robin source and destination
            source_idx = packets_sent % len(nodes)
            dest_idx = (source_idx + 1) % len(nodes)
            
            source_node = nodes[source_idx]
            dest_node = nodes[dest_idx]
            
            # Log and send the packet
            packet_info = {
                'packet_id': packet_id,
                'source': source_node.name,
                'destination': dest_node.name,
                'sent_time': current_time,
                'action': 'sent',
                'delivered': False,
                'with_crdt': with_crdt
            }
            
            source_node.send_packet_to(
                dest_node.name, 
                {
                    'test_data': f"Stress packet {packets_sent}",
                    'packet_id': packet_id,
                    'creation_time': current_time,
                    'size': 1024  # Simulated packet size in bytes
                }
            )
            
            # Record the packet
            metrics.log_packet(packet_info)
            sent_packets.append(packet_info)
            
            packets_sent += 1
            next_packet_time = start_time + (packets_sent * packet_interval)
            
            # Log progress every 100 packets
            if packets_sent % 100 == 0:
                info(f"*** Sent {packets_sent}/{total_packets} packets ***\n")
        
        # Check for deliveries
        for packet in sent_packets:
            if not packet['delivered']:
                dest_node = next((n for n in nodes if n.name == packet['destination']), None)
                if not dest_node:
                    continue
                    
                # Check if destination received the packet
                for bundle_id, bundle in dest_node.bundle_store.items():
                    if isinstance(bundle, dict) and 'packet' in bundle:
                        bundle_packet = bundle['packet']
                        if (hasattr(bundle_packet, 'payload') and 
                            isinstance(bundle_packet.payload, dict) and
                            bundle_packet.payload.get('packet_id') == packet['packet_id']):
                            
                            # Mark as delivered
                            packet['delivered'] = True
                            packet['delivery_time'] = current_time
                            packet['delay'] = current_time - packet['sent_time']
                            
                            # Update metrics
                            metrics.log_packet(packet)
                            break
        
        # Track resource usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        metrics.resource_usage.append((current_time - start_time, memory_usage))
        
        # Calculate protocol overhead
        if with_crdt:
            crdt_size = sum(
                len(json.dumps(node.network_state.to_dict())) + 
                len(json.dumps(node.received_packets.elements))
                for node in nodes
            )
        else:
            crdt_size = 0
            
        metrics.crdt_overhead.append(crdt_size)
        
        # Short sleep to prevent CPU thrashing
        time.sleep(0.01)
    
    # Final delivery check
    for packet in sent_packets:
        if not packet['delivered']:
            dest_node = next((n for n in nodes if n.name == packet['destination']), None)
            if not dest_node:
                continue
                
            # Final check if destination received the packet
            for bundle_id, bundle in dest_node.bundle_store.items():
                if isinstance(bundle, dict) and 'packet' in bundle:
                    bundle_packet = bundle['packet']
                    if (hasattr(bundle_packet, 'payload') and 
                        isinstance(bundle_packet.payload, dict) and
                        bundle_packet.payload.get('packet_id') == packet['packet_id']):
                        
                        # Mark as delivered
                        packet['delivered'] = True
                        packet['delivery_time'] = time.time()
                        packet['delay'] = packet['delivery_time'] - packet['sent_time']
                        
                        # Update metrics
                        metrics.log_packet(packet)
                        break
    
    # Calculate final metrics
    metrics.end_experiment()
    metrics.calculate_metrics()
    
    # Save data
    metrics.save_to_csv(name_suffix)
    
    # Log results
    info(f"\n*** Stress test completed: {name_suffix} ***\n")
    info(f"Total packets sent: {packets_sent}\n")
    info(f"Packets delivered: {sum(1 for p in sent_packets if p['delivered'])}\n")
    info(f"Delivery rate: {metrics.delivery_rates[-1][1]*100:.2f}%\n")
    info(f"Average delay: {np.mean(metrics.end_to_end_delays) if metrics.end_to_end_delays else 0:.2f}s\n")
    
    return metrics


def create_mobility_comparison_summary(results):
    """Create summary comparison across mobility patterns"""
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data for comparison
    patterns = list(results.keys())
    delivery_rates_crdt = [results[p]["with_crdt"].delivery_rates[-1][1]*100 if results[p]["with_crdt"].delivery_rates else 0 for p in patterns]
    delivery_rates_no_crdt = [results[p]["without_crdt"].delivery_rates[-1][1]*100 if results[p]["without_crdt"].delivery_rates else 0 for p in patterns]
    
    delays_crdt = [np.mean(results[p]["with_crdt"].end_to_end_delays) if results[p]["with_crdt"].end_to_end_delays else 0 for p in patterns]
    delays_no_crdt = [np.mean(results[p]["without_crdt"].end_to_end_delays) if results[p]["without_crdt"].end_to_end_delays else 0 for p in patterns]
    
    throughput_crdt = [results[p]["with_crdt"].throughput for p in patterns]
    throughput_no_crdt = [results[p]["without_crdt"].throughput for p in patterns]
    
    overhead_crdt = [np.mean(results[p]["with_crdt"].crdt_overhead) if results[p]["with_crdt"].crdt_overhead else 0 for p in patterns]
    overhead_no_crdt = [np.mean(results[p]["without_crdt"].crdt_overhead) if results[p]["without_crdt"].crdt_overhead else 0 for p in patterns]
    
    # Plot delivery rates
    plt.figure(figsize=(12, 6))
    x = np.arange(len(patterns))
    width = 0.35
    
    plt.bar(x - width/2, delivery_rates_crdt, width, label='With CRDT')
    plt.bar(x + width/2, delivery_rates_no_crdt, width, label='Without CRDT')
    
    plt.xlabel('Mobility Pattern')
    plt.ylabel('Delivery Rate (%)')
    plt.title('Delivery Rate Comparison Across Mobility Patterns')
    plt.xticks(x, patterns)
    plt.legend()
    
    plt.savefig(f'{plots_dir}/mobility_delivery_rates.png')
    plt.close()
    
    # Plot delays
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, delays_crdt, width, label='With CRDT')
    plt.bar(x + width/2, delays_no_crdt, width, label='Without CRDT')
    
    plt.xlabel('Mobility Pattern')
    plt.ylabel('Average Delay (s)')
    plt.title('End-to-End Delay Comparison Across Mobility Patterns')
    plt.xticks(x, patterns)
    plt.legend()
    
    plt.savefig(f'{plots_dir}/mobility_delays.png')
    plt.close()
    
    # Plot throughput
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, throughput_crdt, width, label='With CRDT')
    plt.bar(x + width/2, throughput_no_crdt, width, label='Without CRDT')
    
    plt.xlabel('Mobility Pattern')
    plt.ylabel('Throughput (packets/s)')
    plt.title('Throughput Comparison Across Mobility Patterns')
    plt.xticks(x, patterns)
    plt.legend()
    
    plt.savefig(f'{plots_dir}/mobility_throughput.png')
    plt.close()
    
    # Plot overhead
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, overhead_crdt, width, label='With CRDT')
    plt.bar(x + width/2, overhead_no_crdt, width, label='Without CRDT')
    
    plt.xlabel('Mobility Pattern')
    plt.ylabel('Average Overhead (bytes)')
    plt.title('Protocol Overhead Comparison Across Mobility Patterns')
    plt.xticks(x, patterns)
    plt.legend()
    
    plt.savefig(f'{plots_dir}/mobility_overhead.png')
    plt.close()
    
    # Save summary data
    summary = {
        "mobility_patterns": patterns,
        "delivery_rates": {
            "with_crdt": delivery_rates_crdt,
            "without_crdt": delivery_rates_no_crdt
        },
        "delays": {
            "with_crdt": delays_crdt,
            "without_crdt": delays_no_crdt
        },
        "throughput": {
            "with_crdt": throughput_crdt,
            "without_crdt": throughput_no_crdt
        },
        "overhead": {
            "with_crdt": overhead_crdt,
            "without_crdt": overhead_no_crdt
        }
    }
    
    with open("results/mobility_comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


def create_density_comparison_summary(results):
    """Create summary comparison across network densities"""
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data for comparison
    densities = list(results.keys())
    delivery_rates_crdt = [results[d]["with_crdt"].delivery_rates[-1][1]*100 if results[d]["with_crdt"].delivery_rates else 0 for d in densities]
    delivery_rates_no_crdt = [results[d]["without_crdt"].delivery_rates[-1][1]*100 if results[d]["without_crdt"].delivery_rates else 0 for d in densities]
    
    delays_crdt = [np.mean(results[d]["with_crdt"].end_to_end_delays) if results[d]["with_crdt"].end_to_end_delays else 0 for d in densities]
    delays_no_crdt = [np.mean(results[d]["without_crdt"].end_to_end_delays) if results[d]["without_crdt"].end_to_end_delays else 0 for d in densities]
    
    # Plot delivery rates
    plt.figure(figsize=(12, 6))
    x = np.arange(len(densities))
    width = 0.35
    
    plt.bar(x - width/2, delivery_rates_crdt, width, label='With CRDT')
    plt.bar(x + width/2, delivery_rates_no_crdt, width, label='Without CRDT')
    
    plt.xlabel('Network Density')
    plt.ylabel('Delivery Rate (%)')
    plt.title('Delivery Rate Comparison Across Network Densities')
    plt.xticks(x, densities)
    plt.legend()
    
    plt.savefig(f'{plots_dir}/density_delivery_rates.png')
    plt.close()
    
    # Plot delays
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, delays_crdt, width, label='With CRDT')
    plt.bar(x + width/2, delays_no_crdt, width, label='Without CRDT')
    
    plt.xlabel('Network Density')
    plt.ylabel('Average Delay (s)')
    plt.title('End-to-End Delay Comparison Across Network Densities')
    plt.xticks(x, densities)
    plt.legend()
    
    plt.savefig(f'{plots_dir}/density_delays.png')
    plt.close()
    
    # Save summary data
    summary = {
        "densities": densities,
        "delivery_rates": {
            "with_crdt": delivery_rates_crdt,
            "without_crdt": delivery_rates_no_crdt
        },
        "delays": {
            "with_crdt": delays_crdt,
            "without_crdt": delays_no_crdt
        }
    }
    
    with open("results/density_comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


def create_load_comparison_summary(results):
    """Create summary comparison across network loads"""
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data for comparison
    loads = list(results.keys())
    delivery_rates_crdt = [results[l]["with_crdt"].delivery_rates[-1][1]*100 if results[l]["with_crdt"].delivery_rates else 0 for l in loads]
    delivery_rates_no_crdt = [results[l]["without_crdt"].delivery_rates[-1][1]*100 if results[l]["without_crdt"].delivery_rates else 0 for l in loads]
    
    throughput_crdt = [results[l]["with_crdt"].throughput for l in loads]
    throughput_no_crdt = [results[l]["without_crdt"].throughput for l in loads]
    
    goodput_crdt = [results[l]["with_crdt"].goodput for l in loads]
    goodput_no_crdt = [results[l]["without_crdt"].goodput for l in loads]
    
    overhead_crdt = [np.mean(results[l]["with_crdt"].crdt_overhead) if results[l]["with_crdt"].crdt_overhead else 0 for l in loads]
    overhead_no_crdt = [np.mean(results[l]["without_crdt"].crdt_overhead) if results[l]["without_crdt"].crdt_overhead else 0 for l in loads]
    
    # Plot delivery rates
    plt.figure(figsize=(12, 6))
    x = np.arange(len(loads))
    width = 0.35
    
    plt.bar(x - width/2, delivery_rates_crdt, width, label='With CRDT')
    plt.bar(x + width/2, delivery_rates_no_crdt, width, label='Without CRDT')
    
    plt.xlabel('Network Load')
    plt.ylabel('Delivery Rate (%)')
    plt.title('Delivery Rate Comparison Across Network Loads')
    plt.xticks(x, loads)
    plt.legend()
    
    plt.savefig(f'{plots_dir}/load_delivery_rates.png')
    plt.close()
    
    # Plot throughput vs goodput
    plt.figure(figsize=(12, 10))
    
    # First subplot for throughput
    plt.subplot(2, 1, 1)
    plt.bar(x - width/2, throughput_crdt, width, label='With CRDT')
    plt.bar(x + width/2, throughput_no_crdt, width, label='Without CRDT')
    plt.xlabel('Network Load')
    plt.ylabel('Throughput (packets/s)')
    plt.title('Throughput Comparison Across Network Loads')
    plt.xticks(x, loads)
    plt.legend()
    
    # Second subplot for goodput
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, goodput_crdt, width, label='With CRDT')
    plt.bar(x + width/2, goodput_no_crdt, width, label='Without CRDT')
    plt.xlabel('Network Load')
    plt.ylabel('Goodput (delivered packets/s)')
    plt.title('Goodput Comparison Across Network Loads')
    plt.xticks(x, loads)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/load_throughput_goodput.png')
    plt.close()
    
    # Plot overhead
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, overhead_crdt, width, label='With CRDT')
    plt.bar(x + width/2, overhead_no_crdt, width, label='Without CRDT')
    
    plt.xlabel('Network Load')
    plt.ylabel('Average Overhead (bytes)')
    plt.title('Protocol Overhead Comparison Across Network Loads')
    plt.xticks(x, loads)
    plt.legend()
    
    plt.savefig(f'{plots_dir}/load_overhead.png')
    plt.close()
    
    # Calculate efficiency (goodput/throughput ratio)
    efficiency_crdt = [g/t if t > 0 else 0 for g, t in zip(goodput_crdt, throughput_crdt)]
    efficiency_no_crdt = [g/t if t > 0 else 0 for g, t in zip(goodput_no_crdt, throughput_no_crdt)]
    
    # Plot efficiency
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, [e*100 for e in efficiency_crdt], width, label='With CRDT')
    plt.bar(x + width/2, [e*100 for e in efficiency_no_crdt], width, label='Without CRDT')
    
    plt.xlabel('Network Load')
    plt.ylabel('Efficiency (Goodput/Throughput %) ')
    plt.title('Network Efficiency Comparison Across Network Loads')
    plt.xticks(x, loads)
    plt.legend()
    
    plt.savefig(f'{plots_dir}/load_efficiency.png')
    plt.close()
    
    # Save summary data
    summary = {
        "loads": loads,
        "delivery_rates": {
            "with_crdt": delivery_rates_crdt,
            "without_crdt": delivery_rates_no_crdt
        },
        "throughput": {
            "with_crdt": throughput_crdt,
            "without_crdt": throughput_no_crdt
        },
        "goodput": {
            "with_crdt": goodput_crdt,
            "without_crdt": goodput_no_crdt
        },
        "efficiency": {
            "with_crdt": efficiency_crdt,
            "without_crdt": efficiency_no_crdt
        },
        "overhead": {
            "with_crdt": overhead_crdt,
            "without_crdt": overhead_no_crdt
        }
    }
    
    with open("results/load_comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

def run_reliability_experiment(net, duration=300, packet_loss_rates=[0, 0.1, 0.3, 0.5]):
    """Test performance with different levels of simulated packet loss"""
    # Dictionary to store results for each loss rate
    all_results = {}
    
    for loss_rate in packet_loss_rates:
        loss_name = f"loss_{int(loss_rate*100)}"
        info(f"\n*** Testing with packet loss rate: {loss_rate*100}% ***\n")
        
        # Configure simulated packet loss in nodes
        for i in range(1, 4):  # Assuming 3 nodes
            node = net.get(f'sta{i}')
            node.packet_loss_rate = loss_rate
            info(f"Set packet loss rate for {node.name} to {loss_rate*100}%\n")
        
        # Run with CRDT
        info(f"\n*** Running with CRDT under {loss_name} ***\n")
        crdt_metrics = run_comparative_experiment(
            net, 
            duration=duration, 
            with_crdt=True, 
            name_suffix=loss_name
        )
        
        # Run without CRDT
        info(f"\n*** Running without CRDT under {loss_name} ***\n")
        no_crdt_metrics = run_comparative_experiment(
            net, 
            duration=duration, 
            with_crdt=False, 
            name_suffix=loss_name
        )
        
        # Plot comparative results
        crdt_metrics.plot_comparative_metrics(
            crdt_metrics, 
            no_crdt_metrics, 
            title_suffix=f"({loss_rate*100}% Loss)"
        )
        
        # Store results
        all_results[loss_name] = {
            "with_crdt": crdt_metrics,
            "without_crdt": no_crdt_metrics
        }
    
    # Create summary comparison across loss rates
    create_reliability_comparison_summary(all_results)
    
    return all_results

def create_reliability_comparison_summary(results):
    """Create summary comparison across packet loss rates"""
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data for comparison
    loss_rates = list(results.keys())
    delivery_rates_crdt = [results[l]["with_crdt"].delivery_rates[-1][1]*100 if results[l]["with_crdt"].delivery_rates else 0 for l in loss_rates]
    delivery_rates_no_crdt = [results[l]["without_crdt"].delivery_rates[-1][1]*100 if results[l]["without_crdt"].delivery_rates else 0 for l in loss_rates]
    
    # Calculate resilience (how much delivery rate decreases with increased loss)
    # First get loss percentages from names
    loss_pcts = [int(l.split('_')[1]) for l in loss_rates]
    
    # Sort data by loss percentage
    sorted_indices = np.argsort(loss_pcts)
    loss_pcts = [loss_pcts[i] for i in sorted_indices]
    delivery_rates_crdt = [delivery_rates_crdt[i] for i in sorted_indices]
    delivery_rates_no_crdt = [delivery_rates_no_crdt[i] for i in sorted_indices]
    
    # Plot delivery rates vs loss rate
    plt.figure(figsize=(12, 6))
    plt.plot(loss_pcts, delivery_rates_crdt, 'o-', label='With CRDT')
    plt.plot(loss_pcts, delivery_rates_no_crdt, 's-', label='Without CRDT')
    
    plt.xlabel('Packet Loss Rate (%)')
    plt.ylabel('Delivery Rate (%)')
    plt.title('Delivery Rate Resilience to Packet Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{plots_dir}/reliability_delivery_rates.png')
    plt.close()
    
    # Calculate relative advantage of CRDT at each loss rate
    relative_advantage = [c - n for c, n in zip(delivery_rates_crdt, delivery_rates_no_crdt)]
    
    # Plot CRDT advantage
    plt.figure(figsize=(12, 6))
    plt.bar(loss_pcts, relative_advantage)
    
    plt.xlabel('Packet Loss Rate (%)')
    plt.ylabel('Delivery Rate Advantage (%)')
    plt.title('CRDT Delivery Rate Advantage vs Packet Loss')
    plt.grid(True)
    
    plt.savefig(f'{plots_dir}/reliability_crdt_advantage.png')
    plt.close()
    
    # Save summary data
    summary = {
        "loss_rates": loss_pcts,
        "delivery_rates": {
            "with_crdt": delivery_rates_crdt,
            "without_crdt": delivery_rates_no_crdt
        },
        "crdt_advantage": relative_advantage
    }
    
    with open("results/reliability_comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

def run_latency_experiment(net, duration=300):
    """Test performance with different simulated network latencies"""
    # Dictionary to store results for each latency level
    all_results = {}
    
    # Define different latency levels
    latency_levels = [
        ("low_latency", 10),    # 10ms
        ("medium_latency", 50),  # 50ms
        ("high_latency", 200)    # 200ms
    ]
    
    for latency_name, latency_ms in latency_levels:
        info(f"\n*** Testing with network latency: {latency_ms}ms ***\n")
        
        # Configure simulated latency in nodes
        for i in range(1, 4):  # Assuming 3 nodes
            node = net.get(f'sta{i}')
            node.latency_ms = latency_ms
            info(f"Set network latency for {node.name} to {latency_ms}ms\n")
        
        # Run with CRDT
        info(f"\n*** Running with CRDT under {latency_name} ***\n")
        crdt_metrics = run_comparative_experiment(
            net, 
            duration=duration, 
            with_crdt=True, 
            name_suffix=latency_name
        )
        
        # Run without CRDT
        info(f"\n*** Running without CRDT under {latency_name} ***\n")
        no_crdt_metrics = run_comparative_experiment(
            net, 
            duration=duration, 
            with_crdt=False, 
            name_suffix=latency_name
        )
        
        # Plot comparative results
        crdt_metrics.plot_comparative_metrics(
            crdt_metrics, 
            no_crdt_metrics, 
            title_suffix=f"({latency_ms}ms Latency)"
        )
        
        # Store results
        all_results[latency_name] = {
            "with_crdt": crdt_metrics,
            "without_crdt": no_crdt_metrics
        }
    
    # Create summary comparison across latencies
    create_latency_comparison_summary(all_results, latency_levels)
    
    return all_results

def create_latency_comparison_summary(results, latency_levels):
    """Create summary comparison across network latencies"""
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract latency values for x-axis
    latency_names = [l[0] for l in latency_levels]
    latency_values = [l[1] for l in latency_levels]
    
    # Extract data for comparison
    delivery_rates_crdt = [results[l[0]]["with_crdt"].delivery_rates[-1][1]*100 if results[l[0]]["with_crdt"].delivery_rates else 0 for l in latency_levels]
    delivery_rates_no_crdt = [results[l[0]]["without_crdt"].delivery_rates[-1][1]*100 if results[l[0]]["without_crdt"].delivery_rates else 0 for l in latency_levels]
    
    delay_crdt = [np.mean(results[l[0]]["with_crdt"].end_to_end_delays) if results[l[0]]["with_crdt"].end_to_end_delays else 0 for l in latency_levels]
    delay_no_crdt = [np.mean(results[l[0]]["without_crdt"].end_to_end_delays) if results[l[0]]["without_crdt"].end_to_end_delays else 0 for l in latency_levels]
    
    # Plot delivery rates vs latency
    plt.figure(figsize=(12, 6))
    plt.plot(latency_values, delivery_rates_crdt, 'o-', label='With CRDT')
    plt.plot(latency_values, delivery_rates_no_crdt, 's-', label='Without CRDT')
    
    plt.xlabel('Network Latency (ms)')
    plt.ylabel('Delivery Rate (%)')
    plt.title('Delivery Rate vs Network Latency')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{plots_dir}/latency_delivery_rates.png')
    plt.close()
    
    # Plot end-to-end delay vs latency
    plt.figure(figsize=(12, 6))
    plt.plot(latency_values, delay_crdt, 'o-', label='With CRDT')
    plt.plot(latency_values, delay_no_crdt, 's-', label='Without CRDT')
    
    plt.xlabel('Network Latency (ms)')
    plt.ylabel('End-to-End Delay (s)')
    plt.title('End-to-End Delay vs Network Latency')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{plots_dir}/latency_e2e_delay.png')
    plt.close()
    
    # Save summary data
    summary = {
        "latency_values_ms": latency_values,
        "delivery_rates": {
            "with_crdt": delivery_rates_crdt,
            "without_crdt": delivery_rates_no_crdt
        },
        "end_to_end_delay": {
            "with_crdt": delay_crdt,
            "without_crdt": delay_no_crdt
        }
    }
    
    with open("results/latency_comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

def run_comprehensive_benchmark(net, duration=300):
    """Run a complete benchmark suite for the paper"""
    results_dir = "results/benchmark"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Basic comparison
    info("\n\n*** RUNNING BASIC COMPARISON BENCHMARK ***\n\n")
    crdt_metrics = run_comparative_experiment(
        net, 
        duration=duration, 
        with_crdt=True, 
        name_suffix="baseline"
    )
    
    no_crdt_metrics = run_comparative_experiment(
        net, 
        duration=duration, 
        with_crdt=False, 
        name_suffix="baseline"
    )
    
    crdt_metrics.plot_comparative_metrics(
        crdt_metrics, 
        no_crdt_metrics, 
        title_suffix="(Baseline)"
    )
    
    # 2. Mobility pattern comparison
    info("\n\n*** RUNNING MOBILITY PATTERN BENCHMARK ***\n\n")
    mobility_results = run_mobility_pattern_experiment(net, duration=duration)
    
    # 3. Network density comparison
    info("\n\n*** RUNNING NETWORK DENSITY BENCHMARK ***\n\n")
    density_results = run_density_experiment(net, duration=duration)
    
    # 4. Network load comparison
    info("\n\n*** RUNNING NETWORK LOAD BENCHMARK ***\n\n")
    load_results = run_network_stress_test(net, duration=duration)
    
    # 5. Reliability (packet loss) comparison
    info("\n\n*** RUNNING RELIABILITY BENCHMARK ***\n\n")
    reliability_results = run_reliability_experiment(net, duration=duration)
    
    # 6. Latency comparison
    info("\n\n*** RUNNING LATENCY BENCHMARK ***\n\n")
    latency_results = run_latency_experiment(net, duration=duration)
    
    # Create comprehensive summary
    create_comprehensive_summary(
        baseline=(crdt_metrics, no_crdt_metrics),
        mobility=mobility_results,
        density=density_results,
        load=load_results,
        reliability=reliability_results,
        latency=latency_results
    )
    
    info("\n\n*** BENCHMARKING COMPLETE ***\n\n")
    info(f"Results saved to {results_dir}\n")

def create_comprehensive_summary(baseline, mobility, density, load, reliability, latency):
    """Create a comprehensive summary of all benchmarks"""
    summary = {
        "baseline": {
            "with_crdt": {
                "delivery_rate": baseline[0].delivery_rates[-1][1] if baseline[0].delivery_rates else 0,
                "throughput": baseline[0].throughput,
                "goodput": baseline[0].goodput,
                "avg_delay": np.mean(baseline[0].end_to_end_delays) if baseline[0].end_to_end_delays else 0,
                "missing_packets": baseline[0].missing_packets
            },
            "without_crdt": {
                "delivery_rate": baseline[1].delivery_rates[-1][1] if baseline[1].delivery_rates else 0,
                "throughput": baseline[1].throughput,
                "goodput": baseline[1].goodput,
                "avg_delay": np.mean(baseline[1].end_to_end_delays) if baseline[1].end_to_end_delays else 0,
                "missing_packets": baseline[1].missing_packets
            }
        },
        # Include summaries from other tests
        "improvement_percentage": {}
    }
    
    # Calculate improvement percentage for key metrics
    if baseline[1].throughput > 0:
        summary["improvement_percentage"]["throughput"] = (
            (baseline[0].throughput - baseline[1].throughput) / baseline[1].throughput * 100
        )
    else:
        summary["improvement_percentage"]["throughput"] = 0
        
    if baseline[1].goodput > 0:
        summary["improvement_percentage"]["goodput"] = (
            (baseline[0].goodput - baseline[1].goodput) / baseline[1].goodput * 100
        )
    else:
        summary["improvement_percentage"]["goodput"] = 0
        
    if baseline[1].delivery_rates and baseline[1].delivery_rates[-1][1] > 0:
        baseline_delivery_crdt = baseline[0].delivery_rates[-1][1] if baseline[0].delivery_rates else 0
        baseline_delivery_no_crdt = baseline[1].delivery_rates[-1][1] if baseline[1].delivery_rates else 0
        summary["improvement_percentage"]["delivery_rate"] = (
            (baseline_delivery_crdt - baseline_delivery_no_crdt) / baseline_delivery_no_crdt * 100
        )
    else:
        summary["improvement_percentage"]["delivery_rate"] = 0
    
    # Save comprehensive summary
    with open("results/comprehensive_benchmark_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create a summary plot of CRDT improvement across all experiments
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    metrics = ['delivery_rate', 'throughput', 'goodput']
    improvements = [summary["improvement_percentage"][m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if i > 0 else 'red' for i in improvements]
    plt.bar(metrics, improvements, color=colors)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Metric')
    plt.ylabel('Improvement with CRDT (%)')
    plt.title('CRDT Performance Improvement Summary')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(improvements):
        plt.text(i, v + (5 if v >= 0 else -10), f"{v:.1f}%", 
                 ha='center', va='bottom' if v >= 0 else 'top')
    
    plt.savefig(f'{plots_dir}/overall_improvement_summary.png')
    plt.close()

# Main function to run all experiments
def run_all_experiments(net, duration=300):
    """Run all experimental tests"""
    info("\n*** Starting experimental test suite ***\n")
    
    # 1. Basic comparison
    info("\n*** Running basic comparison test ***\n")
    crdt_metrics = run_comparative_experiment(
        net, 
        duration=duration, 
        with_crdt=True
    )
    
    no_crdt_metrics = run_comparative_experiment(
        net, 
        duration=duration, 
        with_crdt=False
    )
    
    crdt_metrics.plot_comparative_metrics(crdt_metrics, no_crdt_metrics)
    
    # 2. Network density experiment
    info("\n*** Running network density experiment ***\n")
    density_results = run_density_experiment(net, duration=duration)
    
    # 3. Network load experiment
    info("\n*** Running network load experiment ***\n")
    load_results = run_network_stress_test(net, duration=duration)
    
    # 4. Mobility pattern experiment
    info("\n*** Running mobility pattern experiment ***\n")
    mobility_results = run_mobility_pattern_experiment(net, duration=duration)
    
    info("\n*** All experiments completed ***\n")
    info("Results available in the 'results' directory\n") 