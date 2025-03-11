import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from mininet.log import info
from mn_wifi.crdt import GCounter, OrSet

# Set up seaborn for better visualization
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class PerformanceTest:
    """Focused performance testing for opportunistic networking"""
    
    def __init__(self, net, results_dir='results/performance_test'):
        self.net = net
        self.results_dir = results_dir
        self.figures_dir = f"{results_dir}/figures"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            "with_crdt": {},
            "without_crdt": {}
        }
        
        # Test configurations
        self.test_configs = {
            "packet_rates": [5, 10, 20, 50],  # packets per second
            "network_sizes": [3, 5, 7],        # number of nodes
            "mobility_speeds": [1, 3, 5],      # m/s
            "packet_loss_rates": [0, 0.1, 0.3, 0.5]  # percentage as decimal
        }
    
    def run_all_tests(self, duration=120):
        """Run all performance tests"""
        info("\n*** Starting Performance Tests ***\n")
        
        # 1. Test with different packet rates
        self.test_packet_rates(duration)
        
        # 2. Test with different packet loss rates
        self.test_packet_loss(duration)
        
        # 3. Test with different mobility speeds
        self.test_mobility_speeds(duration)
        
        # Generate summary figures
        self.generate_summary_figures()
        
        info("\n*** Performance Tests Completed ***\n")
        info(f"Results saved to {self.results_dir}\n")
    
    def test_packet_rates(self, duration=120):
        """Test performance with different packet rates"""
        info("\n*** Testing Different Packet Rates ***\n")
        
        results_with_crdt = []
        results_without_crdt = []
        
        for rate in self.test_configs["packet_rates"]:
            info(f"\n*** Testing packet rate: {rate} packets/sec ***\n")
            
            # Test with CRDT
            with_crdt_metrics = self.run_single_test(
                duration=duration,
                with_crdt=True,
                packets_per_second=rate,
                name_suffix=f"rate_{rate}"
            )
            results_with_crdt.append({
                "packet_rate": rate,
                "delivery_rate": with_crdt_metrics["delivery_rate"] * 100,
                "throughput": with_crdt_metrics["throughput"],
                "goodput": with_crdt_metrics["goodput"],
                "avg_delay": with_crdt_metrics["avg_delay"],
                "missing_packets": with_crdt_metrics["missing_packets"]
            })
            
            # Test without CRDT
            without_crdt_metrics = self.run_single_test(
                duration=duration,
                with_crdt=False,
                packets_per_second=rate,
                name_suffix=f"rate_{rate}"
            )
            results_without_crdt.append({
                "packet_rate": rate,
                "delivery_rate": without_crdt_metrics["delivery_rate"] * 100,
                "throughput": without_crdt_metrics["throughput"],
                "goodput": without_crdt_metrics["goodput"],
                "avg_delay": without_crdt_metrics["avg_delay"],
                "missing_packets": without_crdt_metrics["missing_packets"]
            })
        
        # Store results
        self.results["with_crdt"]["packet_rates"] = results_with_crdt
        self.results["without_crdt"]["packet_rates"] = results_without_crdt
        
        # Generate figures
        self.generate_packet_rate_figures()
    
    def test_packet_loss(self, duration=120):
        """Test performance with different packet loss rates"""
        info("\n*** Testing Different Packet Loss Rates ***\n")
        
        results_with_crdt = []
        results_without_crdt = []
        
        for loss_rate in self.test_configs["packet_loss_rates"]:
            info(f"\n*** Testing packet loss rate: {loss_rate*100}% ***\n")
            
            # Configure nodes with packet loss
            for i in range(1, 4):  # Assuming at least 3 nodes
                node = self.net.get(f'sta{i}')
                node.packet_loss_rate = loss_rate
            
            # Test with CRDT
            with_crdt_metrics = self.run_single_test(
                duration=duration,
                with_crdt=True,
                name_suffix=f"loss_{int(loss_rate*100)}"
            )
            results_with_crdt.append({
                "loss_rate": loss_rate*100,
                "delivery_rate": with_crdt_metrics["delivery_rate"] * 100,
                "throughput": with_crdt_metrics["throughput"],
                "goodput": with_crdt_metrics["goodput"],
                "avg_delay": with_crdt_metrics["avg_delay"],
                "missing_packets": with_crdt_metrics["missing_packets"]
            })
            
            # Test without CRDT
            without_crdt_metrics = self.run_single_test(
                duration=duration,
                with_crdt=False,
                name_suffix=f"loss_{int(loss_rate*100)}"
            )
            results_without_crdt.append({
                "loss_rate": loss_rate*100,
                "delivery_rate": without_crdt_metrics["delivery_rate"] * 100,
                "throughput": without_crdt_metrics["throughput"],
                "goodput": without_crdt_metrics["goodput"],
                "avg_delay": without_crdt_metrics["avg_delay"],
                "missing_packets": without_crdt_metrics["missing_packets"]
            })
            
            # Reset packet loss
            for i in range(1, 4):
                node = self.net.get(f'sta{i}')
                node.packet_loss_rate = 0
        
        # Store results
        self.results["with_crdt"]["packet_loss"] = results_with_crdt
        self.results["without_crdt"]["packet_loss"] = results_without_crdt
        
        # Generate figures
        self.generate_packet_loss_figures()
    
    def test_mobility_speeds(self, duration=120):
        """Test performance with different mobility speeds"""
        info("\n*** Testing Different Mobility Speeds ***\n")
        
        results_with_crdt = []
        results_without_crdt = []
        
        for speed in self.test_configs["mobility_speeds"]:
            info(f"\n*** Testing mobility speed: {speed} m/s ***\n")
            
            # Configure mobility model
            self.net.setMobilityModel(
                model="random_waypoint",
                max_x=100,
                max_y=100,
                min_v=speed,
                max_v=speed
            )
            
            # Test with CRDT
            with_crdt_metrics = self.run_single_test(
                duration=duration,
                with_crdt=True,
                name_suffix=f"speed_{speed}"
            )
            results_with_crdt.append({
                "speed": speed,
                "delivery_rate": with_crdt_metrics["delivery_rate"] * 100,
                "throughput": with_crdt_metrics["throughput"],
                "goodput": with_crdt_metrics["goodput"],
                "avg_delay": with_crdt_metrics["avg_delay"],
                "missing_packets": with_crdt_metrics["missing_packets"]
            })
            
            # Test without CRDT
            without_crdt_metrics = self.run_single_test(
                duration=duration,
                with_crdt=False,
                name_suffix=f"speed_{speed}"
            )
            results_without_crdt.append({
                "speed": speed,
                "delivery_rate": without_crdt_metrics["delivery_rate"] * 100,
                "throughput": without_crdt_metrics["throughput"],
                "goodput": without_crdt_metrics["goodput"],
                "avg_delay": without_crdt_metrics["avg_delay"],
                "missing_packets": without_crdt_metrics["missing_packets"]
            })
        
        # Store results
        self.results["with_crdt"]["mobility_speeds"] = results_with_crdt
        self.results["without_crdt"]["mobility_speeds"] = results_without_crdt
        
        # Generate figures
        self.generate_mobility_figures()
    
    def run_single_test(self, duration=120, with_crdt=True, packets_per_second=10, name_suffix=""):
        """Run a single performance test"""
        # Get nodes
        nodes = [self.net.get(f'sta{i+1}') for i in range(3)]  # Assuming at least 3 nodes
        
        # Configure CRDT mode
        if with_crdt:
            for node in nodes:
                node.max_crdt_packets = 9999999
                info(f"*** Enabled CRDT updates for {node.name}\n")
        else:
            for node in nodes:
                node.max_crdt_packets = 0
                info(f"*** Disabled CRDT updates for {node.name}\n")
        
        # Reset node state
        for node in nodes:
            node.bundle_store = {}
            node.network_state = GCounter()
            node.received_packets = OrSet()
            node.packets_sent = 0
            node.crdt_packets_sent = 0
        
        # Test parameters
        source_node = nodes[0]
        total_packets = int(packets_per_second * duration)
        packet_interval = 1.0 / packets_per_second
        
        info(f"\n*** Starting test: {'With CRDT' if with_crdt else 'Without CRDT'} - {name_suffix} ***\n")
        info(f"*** Sending {total_packets} packets at {packets_per_second} packets/sec ***\n")
        
        # Tracking variables
        start_time = time.time()
        next_packet_time = start_time
        packets_sent = 0
        sent_packets = []
        delivered_packets = []
        end_to_end_delays = []
        
        # Main test loop
        while time.time() - start_time < duration and packets_sent < total_packets:
            current_time = time.time()
            
            # Send packets at regular intervals
            if current_time >= next_packet_time and packets_sent < total_packets:
                # Create a test packet
                packet_id = f"test_{name_suffix}_{packets_sent}"
                dest_idx = (packets_sent % (len(nodes) - 1)) + 1  # Round-robin to other nodes
                dest_node = nodes[dest_idx]
                
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
                
                # Track sent packet
                sent_packets.append({
                    'packet_id': packet_id,
                    'source': source_node.name,
                    'destination': dest_node.name,
                    'sent_time': current_time,
                    'delivered': False
                })
                
                packets_sent += 1
                next_packet_time = start_time + (packets_sent * packet_interval)
                
                # Log progress
                if packets_sent % 50 == 0:
                    info(f"*** Sent {packets_sent}/{total_packets} packets ***\n")
            
            # Check for deliveries
            for packet in sent_packets:
                if not packet['delivered']:
                    dest_node = self.net.get(packet['destination'])
                    
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
                                
                                delivered_packets.append(packet)
                                end_to_end_delays.append(packet['delay'])
                                break
            
            # Short sleep to prevent CPU thrashing
            time.sleep(0.01)
        
        # Calculate metrics
        test_duration = time.time() - start_time
        delivery_rate = len(delivered_packets) / len(sent_packets) if sent_packets else 0
        throughput = packets_sent / test_duration if test_duration > 0 else 0
        goodput = len(delivered_packets) / test_duration if test_duration > 0 else 0
        avg_delay = np.mean(end_to_end_delays) if end_to_end_delays else 0
        missing_packets = 100 - (delivery_rate * 100)
        
        # Log results
        info(f"\n*** Test completed: {'With CRDT' if with_crdt else 'Without CRDT'} - {name_suffix} ***\n")
        info(f"Delivery rate: {delivery_rate*100:.2f}%\n")
        info(f"Throughput: {throughput:.2f} packets/sec\n")
        info(f"Goodput: {goodput:.2f} packets/sec\n")
        info(f"Average delay: {avg_delay:.4f} seconds\n")
        info(f"Missing packets: {missing_packets:.2f}%\n")
        
        # Save results
        test_type = "with_crdt" if with_crdt else "without_crdt"
        results = {
            "delivery_rate": delivery_rate,
            "throughput": throughput,
            "goodput": goodput,
            "avg_delay": avg_delay,
            "missing_packets": missing_packets,
            "test_duration": test_duration,
            "packets_sent": packets_sent,
            "packets_delivered": len(delivered_packets)
        }
        
        with open(f"{self.results_dir}/{test_type}_{name_suffix}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def generate_packet_rate_figures(self):
        """Generate figures for packet rate tests"""
        # Convert results to DataFrames
        df_crdt = pd.DataFrame(self.results["with_crdt"]["packet_rates"])
        df_no_crdt = pd.DataFrame(self.results["without_crdt"]["packet_rates"])
        
        # 1. Delivery Rate vs Packet Rate
        plt.figure(figsize=(12, 8))
        plt.plot(df_crdt["packet_rate"], df_crdt["delivery_rate"], 'o-', linewidth=2, markersize=10, label='With CRDT')
        plt.plot(df_no_crdt["packet_rate"], df_no_crdt["delivery_rate"], 's--', linewidth=2, markersize=10, label='Without CRDT')
        plt.xlabel('Packet Rate (packets/sec)', fontsize=14)
        plt.ylabel('Delivery Rate (%)', fontsize=14)
        plt.title('Delivery Rate vs Packet Rate', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/delivery_rate_vs_packet_rate.png", dpi=300)
        plt.close()
        
        # 2. Throughput and Goodput vs Packet Rate
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        ax1.set_xlabel('Packet Rate (packets/sec)', fontsize=14)
        ax1.set_ylabel('Throughput (packets/sec)', fontsize=14, color='tab:blue')
        ax1.plot(df_crdt["packet_rate"], df_crdt["throughput"], 'o-', linewidth=2, markersize=10, color='tab:blue', label='Throughput (CRDT)')
        ax1.plot(df_no_crdt["packet_rate"], df_no_crdt["throughput"], 's--', linewidth=2, markersize=10, color='tab:cyan', label='Throughput (No CRDT)')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Goodput (packets/sec)', fontsize=14, color='tab:red')
        ax2.plot(df_crdt["packet_rate"], df_crdt["goodput"], 'o-', linewidth=2, markersize=10, color='tab:red', label='Goodput (CRDT)')
        ax2.plot(df_no_crdt["packet_rate"], df_no_crdt["goodput"], 's--', linewidth=2, markersize=10, color='tab:orange', label='Goodput (No CRDT)')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
        
        plt.title('Throughput and Goodput vs Packet Rate', fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/throughput_goodput_vs_packet_rate.png", dpi=300)
        plt.close()
        
        # 3. Average Delay vs Packet Rate
        plt.figure(figsize=(12, 8))
        plt.plot(df_crdt["packet_rate"], df_crdt["avg_delay"], 'o-', linewidth=2, markersize=10, label='With CRDT')
        plt.plot(df_no_crdt["packet_rate"], df_no_crdt["avg_delay"], 's--', linewidth=2, markersize=10, label='Without CRDT')
        plt.xlabel('Packet Rate (packets/sec)', fontsize=14)
        plt.ylabel('Average End-to-End Delay (seconds)', fontsize=14)
        plt.title('Average Delay vs Packet Rate', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/avg_delay_vs_packet_rate.png", dpi=300)
        plt.close()
        
        # 4. Missing Packets vs Packet Rate
        plt.figure(figsize=(12, 8))
        plt.plot(df_crdt["packet_rate"], df_crdt["missing_packets"], 'o-', linewidth=2, markersize=10, label='With CRDT')
        plt.plot(df_no_crdt["packet_rate"], df_no_crdt["missing_packets"], 's--', linewidth=2, markersize=10, label='Without CRDT')
        plt.xlabel('Packet Rate (packets/sec)', fontsize=14)
        plt.ylabel('Missing Packets (%)', fontsize=14)
        plt.title('Missing Packets vs Packet Rate', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/missing_packets_vs_packet_rate.png", dpi=300)
        plt.close()
    
    def generate_packet_loss_figures(self):
        """Generate figures for packet loss tests"""
        # Convert results to DataFrames
        df_crdt = pd.DataFrame(self.results["with_crdt"]["packet_loss"])
        df_no_crdt = pd.DataFrame(self.results["without_crdt"]["packet_loss"])
        
        # 1. Delivery Rate vs Packet Loss
        plt.figure(figsize=(12, 8))
        plt.plot(df_crdt["loss_rate"], df_crdt["delivery_rate"], 'o-', linewidth=2, markersize=10, label='With CRDT')
        plt.plot(df_no_crdt["loss_rate"], df_no_crdt["delivery_rate"], 's--', linewidth=2, markersize=10, label='Without CRDT')
        plt.xlabel('Packet Loss Rate (%)', fontsize=14)
        plt.ylabel('Delivery Rate (%)', fontsize=14)
        plt.title('Delivery Rate vs Packet Loss', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/delivery_rate_vs_packet_loss.png", dpi=300)
        plt.close()
        
        # 2. Throughput and Goodput vs Packet Loss
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        ax1.set_xlabel('Packet Loss Rate (%)', fontsize=14)
        ax1.set_ylabel('Throughput (packets/sec)', fontsize=14, color='tab:blue')
        ax1.plot(df_crdt["loss_rate"], df_crdt["throughput"], 'o-', linewidth=2, markersize=10, color='tab:blue', label='Throughput (CRDT)')
        ax1.plot(df_no_crdt["loss_rate"], df_no_crdt["throughput"], 's--', linewidth=2, markersize=10, color='tab:cyan', label='Throughput (No CRDT)')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Goodput (packets/sec)', fontsize=14, color='tab:red')
        ax2.plot(df_crdt["loss_rate"], df_crdt["goodput"], 'o-', linewidth=2, markersize=10, color='tab:red', label='Goodput (CRDT)')
        ax2.plot(df_no_crdt["loss_rate"], df_no_crdt["goodput"], 's--', linewidth=2, markersize=10, color='tab:orange', label='Goodput (No CRDT)')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
        
        plt.title('Throughput and Goodput vs Packet Loss', fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/throughput_goodput_vs_packet_loss.png", dpi=300)
        plt.close()
        
        # 3. Average Delay vs Packet Loss
        plt.figure(figsize=(12, 8))
        plt.plot(df_crdt["loss_rate"], df_crdt["avg_delay"], 'o-', linewidth=2, markersize=10, label='With CRDT')
        plt.plot(df_no_crdt["loss_rate"], df_no_crdt["avg_delay"], 's--', linewidth=2, markersize=10, label='Without CRDT')
        plt.xlabel('Packet Loss Rate (%)', fontsize=14)
        plt.ylabel('Average End-to-End Delay (seconds)', fontsize=14)
        plt.title('Average Delay vs Packet Loss', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/avg_delay_vs_packet_loss.png", dpi=300)
        plt.close()
    
    def generate_mobility_figures(self):
        """Generate figures for mobility tests"""
        # Convert results to DataFrames
        df_crdt = pd.DataFrame(self.results["with_crdt"]["mobility_speeds"])
        df_no_crdt = pd.DataFrame(self.results["without_crdt"]["mobility_speeds"])
        
        # 1. Delivery Rate vs Mobility Speed
        plt.figure(figsize=(12, 8))
        plt.plot(df_crdt["speed"], df_crdt["delivery_rate"], 'o-', linewidth=2, markersize=10, label='With CRDT')
        plt.plot(df_no_crdt["speed"], df_no_crdt["delivery_rate"], 's--', linewidth=2, markersize=10, label='Without CRDT')
        plt.xlabel('Mobility Speed (m/s)', fontsize=14)
        plt.ylabel('Delivery Rate (%)', fontsize=14)
        plt.title('Delivery Rate vs Mobility Speed', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/delivery_rate_vs_mobility.png", dpi=300)
        plt.close()
        
        # 2. Average Delay vs Mobility Speed
        plt.figure(figsize=(12, 8))
        plt.plot(df_crdt["speed"], df_crdt["avg_delay"], 'o-', linewidth=2, markersize=10, label='With CRDT')
        plt.plot(df_no_crdt["speed"], df_no_crdt["avg_delay"], 's--', linewidth=2, markersize=10, label='Without CRDT')
        plt.xlabel('Mobility Speed (m/s)', fontsize=14)
        plt.ylabel('Average End-to-End Delay (seconds)', fontsize=14)
        plt.title('Average Delay vs Mobility Speed', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/avg_delay_vs_mobility.png", dpi=300)
        plt.close()
        
        # 3. Missing Packets vs Mobility Speed
        plt.figure(figsize=(12, 8))
        plt.plot(df_crdt["speed"], df_crdt["missing_packets"], 'o-', linewidth=2, markersize=10, label='With CRDT')
        plt.plot(df_no_crdt["speed"], df_no_crdt["missing_packets"], 's--', linewidth=2, markersize=10, label='Without CRDT')
        plt.xlabel('Mobility Speed (m/s)', fontsize=14)
        plt.ylabel('Missing Packets (%)', fontsize=14)
        plt.title('Missing Packets vs Mobility Speed', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/missing_packets_vs_mobility.png", dpi=300)
        plt.close()
    
    def generate_summary_figures(self):
        """Generate summary figures comparing all metrics"""
        # Create a summary dataframe for each test type
        summary_data = []
        
        # Add packet rate data
        for i, rate in enumerate(self.test_configs["packet_rates"]):
            crdt_data = self.results["with_crdt"]["packet_rates"][i]
            no_crdt_data = self.results["without_crdt"]["packet_rates"][i]
            
            summary_data.append({
                "test_type": "Packet Rate",
                "parameter": f"{rate} pkt/s",
                "delivery_rate_crdt": crdt_data["delivery_rate"],
                "delivery_rate_no_crdt": no_crdt_data["delivery_rate"],
                "throughput_crdt": crdt_data["throughput"],
                "throughput_no_crdt": no_crdt_data["throughput"],
                "goodput_crdt": crdt_data["goodput"],
                "goodput_no_crdt": no_crdt_data["goodput"],
                "avg_delay_crdt": crdt_data["avg_delay"],
                "avg_delay_no_crdt": no_crdt_data["avg_delay"],
                "missing_packets_crdt": crdt_data["missing_packets"],
                "missing_packets_no_crdt": no_crdt_data["missing_packets"]
            })
        
        # Add packet loss data
        for i, loss in enumerate(self.test_configs["packet_loss_rates"]):
            crdt_data = self.results["with_crdt"]["packet_loss"][i]
            no_crdt_data = self.results["without_crdt"]["packet_loss"][i]
            
            summary_data.append({
                "test_type": "Packet Loss",
                "parameter": f"{loss*100}%",
                "delivery_rate_crdt": crdt_data["delivery_rate"],
                "delivery_rate_no_crdt": no_crdt_data["delivery_rate"],
                "throughput_crdt": crdt_data["throughput"],
                "throughput_no_crdt": no_crdt_data["throughput"],
                "goodput_crdt": crdt_data["goodput"],
                "goodput_no_crdt": no_crdt_data["goodput"],
                "avg_delay_crdt": crdt_data["avg_delay"],
                "avg_delay_no_crdt": no_crdt_data["avg_delay"],
                "missing_packets_crdt": crdt_data["missing_packets"],
                "missing_packets_no_crdt": no_crdt_data["missing_packets"]
            })
        
        # Add mobility data
        for i, speed in enumerate(self.test_configs["mobility_speeds"]):
            crdt_data = self.results["with_crdt"]["mobility_speeds"][i]
            no_crdt_data = self.results["without_crdt"]["mobility_speeds"][i]
            
            summary_data.append({
                "test_type": "Mobility",
                "parameter": f"{speed} m/s",
                "delivery_rate_crdt": crdt_data["delivery_rate"],
                "delivery_rate_no_crdt": no_crdt_data["delivery_rate"],
                "throughput_crdt": crdt_data["throughput"],
                "throughput_no_crdt": no_crdt_data["throughput"],
                "goodput_crdt": crdt_data["goodput"],
                "goodput_no_crdt": no_crdt_data["goodput"],
                "avg_delay_crdt": crdt_data["avg_delay"],
                "avg_delay_no_crdt": no_crdt_data["avg_delay"],
                "missing_packets_crdt": crdt_data["missing_packets"],
                "missing_packets_no_crdt": no_crdt_data["missing_packets"]
            })
        
        # Convert to DataFrame
        df_summary = pd.DataFrame(summary_data)
        
        # Save summary data
        df_summary.to_csv(f"{self.results_dir}/summary_data.csv", index=False)
        
        # Create comprehensive summary figure
        self._create_comprehensive_figure(df_summary)
        
        # Create CRDT improvement figure
        self._create_improvement_figure(df_summary)
    
    def _create_comprehensive_figure(self, df):
        """Create a comprehensive figure showing all metrics"""
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(18, 20))
        
        # Group by test type
        test_types = df["test_type"].unique()
        
        # Color palette
        colors = sns.color_palette("husl", len(test_types))
        
        # 1. Delivery Rate (top left)
        ax = axes[0, 0]
        for i, test_type in enumerate(test_types):
            test_data = df[df["test_type"] == test_type]
            ax.plot(test_data["parameter"], test_data["delivery_rate_crdt"], 'o-', 
                    color=colors[i], label=f"{test_type} (CRDT)")
            ax.plot(test_data["parameter"], test_data["delivery_rate_no_crdt"], 's--', 
                    color=colors[i], alpha=0.6, label=f"{test_type} (No CRDT)")
        
        ax.set_title("Delivery Rate Comparison", fontsize=16)
        ax.set_ylabel("Delivery Rate (%)", fontsize=14)
        ax.set_xlabel("Parameter Value", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=10, loc='best')
        
        # 2. Throughput (top right)
        ax = axes[0, 1]
        for i, test_type in enumerate(test_types):
            test_data = df[df["test_type"] == test_type]
            ax.plot(test_data["parameter"], test_data["throughput_crdt"], 'o-', 
                    color=colors[i], label=f"{test_type} (CRDT)")
            ax.plot(test_data["parameter"], test_data["throughput_no_crdt"], 's--', 
                    color=colors[i], alpha=0.6, label=f"{test_type} (No CRDT)")
        
        ax.set_title("Throughput Comparison", fontsize=16)
        ax.set_ylabel("Throughput (packets/sec)", fontsize=14)
        ax.set_xlabel("Parameter Value", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=10, loc='best')
        
        # 3. Goodput (middle left)
        ax = axes[1, 0]
        for i, test_type in enumerate(test_types):
            test_data = df[df["test_type"] == test_type]
            ax.plot(test_data["parameter"], test_data["goodput_crdt"], 'o-', 
                    color=colors[i], label=f"{test_type} (CRDT)")
            ax.plot(test_data["parameter"], test_data["goodput_no_crdt"], 's--', 
                    color=colors[i], alpha=0.6, label=f"{test_type} (No CRDT)")
        
        ax.set_title("Goodput Comparison", fontsize=16)
        ax.set_ylabel("Goodput (packets/sec)", fontsize=14)
        ax.set_xlabel("Parameter Value", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=10, loc='best')
        
        # 4. Average Delay (middle right)
        ax = axes[1, 1]
        for i, test_type in enumerate(test_types):
            test_data = df[df["test_type"] == test_type]
            ax.plot(test_data["parameter"], test_data["avg_delay_crdt"], 'o-', 
                    color=colors[i], label=f"{test_type} (CRDT)")
            ax.plot(test_data["parameter"], test_data["avg_delay_no_crdt"], 's--', 
                    color=colors[i], alpha=0.6, label=f"{test_type} (No CRDT)")
        
        ax.set_title("Average End-to-End Delay Comparison", fontsize=16)
        ax.set_ylabel("Delay (seconds)", fontsize=14)
        ax.set_xlabel("Parameter Value", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=10, loc='best')
        
        # 5. Missing Packets (bottom left)
        ax = axes[2, 0]
        for i, test_type in enumerate(test_types):
            test_data = df[df["test_type"] == test_type]
            ax.plot(test_data["parameter"], test_data["missing_packets_crdt"], 'o-', 
                    color=colors[i], label=f"{test_type} (CRDT)")
            ax.plot(test_data["parameter"], test_data["missing_packets_no_crdt"], 's--', 
                    color=colors[i], alpha=0.6, label=f"{test_type} (No CRDT)")
        
        ax.set_title("Missing Packets Comparison", fontsize=16)
        ax.set_ylabel("Missing Packets (%)", fontsize=14)
        ax.set_xlabel("Parameter Value", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=10, loc='best')
        
        # 6. Efficiency (Goodput/Throughput) (bottom right)
        ax = axes[2, 1]
        for i, test_type in enumerate(test_types):
            test_data = df[df["test_type"] == test_type]
            efficiency_crdt = test_data["goodput_crdt"] / test_data["throughput_crdt"] * 100
            efficiency_no_crdt = test_data["goodput_no_crdt"] / test_data["throughput_no_crdt"] * 100
            
            ax.plot(test_data["parameter"], efficiency_crdt, 'o-', 
                    color=colors[i], label=f"{test_type} (CRDT)")
            ax.plot(test_data["parameter"], efficiency_no_crdt, 's--', 
                    color=colors[i], alpha=0.6, label=f"{test_type} (No CRDT)")
        
        ax.set_title("Network Efficiency Comparison", fontsize=16)
        ax.set_ylabel("Efficiency (Goodput/Throughput %)", fontsize=14)
        ax.set_xlabel("Parameter Value", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=10, loc='best')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/comprehensive_comparison.png", dpi=300)
        plt.close()
    
    def _create_improvement_figure(self, df):
        """Create a figure showing the improvement from using CRDT"""
        # Calculate improvement percentages
        df["delivery_rate_improvement"] = ((df["delivery_rate_crdt"] - df["delivery_rate_no_crdt"]) / 
                                          df["delivery_rate_no_crdt"] * 100)
        df["throughput_improvement"] = ((df["throughput_crdt"] - df["throughput_no_crdt"]) / 
                                       df["throughput_no_crdt"] * 100)
        df["goodput_improvement"] = ((df["goodput_crdt"] - df["goodput_no_crdt"]) / 
                                    df["goodput_no_crdt"] * 100)
        df["delay_improvement"] = ((df["avg_delay_no_crdt"] - df["avg_delay_crdt"]) / 
                                  df["avg_delay_no_crdt"] * 100)  # Lower delay is better
        df["missing_packets_improvement"] = ((df["missing_packets_no_crdt"] - df["missing_packets_crdt"]) / 
                                           df["missing_packets_no_crdt"] * 100)  # Lower missing is better
        
        # Replace inf and NaN with 0
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        
        # Group by test type
        test_types = df["test_type"].unique()
        
        # 1. Delivery Rate and Throughput Improvement
        ax = axes[0]
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df["delivery_rate_improvement"], width, label='Delivery Rate Improvement')
        ax.bar(x + width/2, df["throughput_improvement"], width, label='Throughput Improvement')
        
        ax.set_title("Delivery Rate and Throughput Improvement with CRDT", fontsize=16)
        ax.set_ylabel("Improvement (%)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['test_type']}: {row['parameter']}" for _, row in df.iterrows()], 
                          rotation=45, ha='right')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, axis='y')
        ax.legend(fontsize=12)
        
        # Add values on top of bars
        for i, v in enumerate(df["delivery_rate_improvement"]):
            ax.text(i - width/2, v + (5 if v >= 0 else -10), f"{v:.1f}%", 
                   ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        
        for i, v in enumerate(df["throughput_improvement"]):
            ax.text(i + width/2, v + (5 if v >= 0 else -10), f"{v:.1f}%", 
                   ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        
        # 2. Goodput and Delay Improvement
        ax = axes[1]
        
        ax.bar(x - width/2, df["goodput_improvement"], width, label='Goodput Improvement')
        ax.bar(x + width/2, df["delay_improvement"], width, label='Delay Improvement')
        
        ax.set_title("Goodput and Delay Improvement with CRDT", fontsize=16)
        ax.set_ylabel("Improvement (%)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['test_type']}: {row['parameter']}" for _, row in df.iterrows()], 
                          rotation=45, ha='right')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, axis='y')
        ax.legend(fontsize=12)
        
        # Add values on top of bars
        for i, v in enumerate(df["goodput_improvement"]):
            ax.text(i - width/2, v + (5 if v >= 0 else -10), f"{v:.1f}%", 
                   ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        
        for i, v in enumerate(df["delay_improvement"]):
            ax.text(i + width/2, v + (5 if v >= 0 else -10), f"{v:.1f}%", 
                   ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        
        # 3. Missing Packets Improvement
        ax = axes[2]
        
        ax.bar(x, df["missing_packets_improvement"], width, label='Missing Packets Improvement')
        
        ax.set_title("Missing Packets Improvement with CRDT", fontsize=16)
        ax.set_ylabel("Improvement (%)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['test_type']}: {row['parameter']}" for _, row in df.iterrows()], 
                          rotation=45, ha='right')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, axis='y')
        ax.legend(fontsize=12)
        
        # Add values on top of bars
        for i, v in enumerate(df["missing_packets_improvement"]):
            ax.text(i, v + (5 if v >= 0 else -10), f"{v:.1f}%", 
                   ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/crdt_improvement.png", dpi=300)
        plt.close()
        
        # Save improvement data
        improvement_data = df[["test_type", "parameter", 
                              "delivery_rate_improvement", "throughput_improvement", 
                              "goodput_improvement", "delay_improvement", 
                              "missing_packets_improvement"]]
        improvement_data.to_csv(f"{self.results_dir}/improvement_data.csv", index=False)


# Function to generate random test data (for testing without running actual experiments)
def generate_random_test_data():
    """Generate random test data for visualization testing"""
    results_dir = 'results/random_test'
    figures_dir = f"{results_dir}/figures"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Test configurations
    packet_rates = [5, 10, 20, 50]
    packet_loss_rates = [0, 10, 30, 50]  # percentages
    mobility_speeds = [1, 3, 5]
    
    # Generate random data
    random_data = {
        "with_crdt": {
            "packet_rates": [],
            "packet_loss": [],
            "mobility_speeds": []
        },
        "without_crdt": {
            "packet_rates": [],
            "packet_loss": [],
            "mobility_speeds": []
        }
    }
    
    # Generate packet rate data
    for rate in packet_rates:
        # With CRDT (generally better performance)
        delivery_rate = min(95 - rate/2 + np.random.normal(0, 5), 100) / 100
        throughput = rate * 0.95 + np.random.normal(0, 1)
        goodput = throughput * delivery_rate
        avg_delay = 0.05 + rate/200 + np.random.normal(0, 0.01)
        missing_packets = (1 - delivery_rate) * 100
        
        random_data["with_crdt"]["packet_rates"].append({
            "packet_rate": rate,
            "delivery_rate": delivery_rate * 100,
            "throughput": throughput,
            "goodput": goodput,
            "avg_delay": avg_delay,
            "missing_packets": missing_packets
        })
        
        # Without CRDT (generally worse performance)
        delivery_rate = min(85 - rate/1.5 + np.random.normal(0, 5), 100) / 100
        throughput = rate * 0.9 + np.random.normal(0, 1)
        goodput = throughput * delivery_rate
        avg_delay = 0.08 + rate/150 + np.random.normal(0, 0.01)
        missing_packets = (1 - delivery_rate) * 100
        
        random_data["without_crdt"]["packet_rates"].append({
            "packet_rate": rate,
            "delivery_rate": delivery_rate * 100,
            "throughput": throughput,
            "goodput": goodput,
            "avg_delay": avg_delay,
            "missing_packets": missing_packets
        })
    
    # Generate packet loss data
    for loss in packet_loss_rates:
        # With CRDT (more resilient to packet loss)
        delivery_rate = min(95 - loss/2 + np.random.normal(0, 3), 100) / 100
        throughput = 20 * (1 - loss/150) + np.random.normal(0, 1)
        goodput = throughput * delivery_rate
        avg_delay = 0.05 + loss/500 + np.random.normal(0, 0.01)
        missing_packets = (1 - delivery_rate) * 100
        
        random_data["with_crdt"]["packet_loss"].append({
            "loss_rate": loss,
            "delivery_rate": delivery_rate * 100,
            "throughput": throughput,
            "goodput": goodput,
            "avg_delay": avg_delay,
            "missing_packets": missing_packets
        })
        
        # Without CRDT (less resilient to packet loss)
        delivery_rate = min(85 - loss/1.2 + np.random.normal(0, 3), 100) / 100
        throughput = 20 * (1 - loss/120) + np.random.normal(0, 1)
        goodput = throughput * delivery_rate
        avg_delay = 0.08 + loss/400 + np.random.normal(0, 0.01)
        missing_packets = (1 - delivery_rate) * 100
        
        random_data["without_crdt"]["packet_loss"].append({
            "loss_rate": loss,
            "delivery_rate": delivery_rate * 100,
            "throughput": throughput,
            "goodput": goodput,
            "avg_delay": avg_delay,
            "missing_packets": missing_packets
        })
    
    # Generate mobility data
    for speed in mobility_speeds:
        # With CRDT (better with mobility)
        delivery_rate = min(95 - speed*3 + np.random.normal(0, 3), 100) / 100
        throughput = 20 * (1 - speed/30) + np.random.normal(0, 1)
        goodput = throughput * delivery_rate
        avg_delay = 0.05 + speed/50 + np.random.normal(0, 0.01)
        missing_packets = (1 - delivery_rate) * 100
        
        random_data["with_crdt"]["mobility_speeds"].append({
            "speed": speed,
            "delivery_rate": delivery_rate * 100,
            "throughput": throughput,
            "goodput": goodput,
            "avg_delay": avg_delay,
            "missing_packets": missing_packets
        })
        
        # Without CRDT (worse with mobility)
        delivery_rate = min(85 - speed*5 + np.random.normal(0, 3), 100) / 100
        throughput = 20 * (1 - speed/25) + np.random.normal(0, 1)
        goodput = throughput * delivery_rate
        avg_delay = 0.08 + speed/40 + np.random.normal(0, 0.01)
        missing_packets = (1 - delivery_rate) * 100
        
        random_data["without_crdt"]["mobility_speeds"].append({
            "speed": speed,
            "delivery_rate": delivery_rate * 100,
            "throughput": throughput,
            "goodput": goodput,
            "avg_delay": avg_delay,
            "missing_packets": missing_packets
        })
    
    # Save random data
    with open(f"{results_dir}/random_data.json", 'w') as f:
        json.dump(random_data, f, indent=2)
    
    # Create test object with the random data
    class RandomTest:
        def __init__(self):
            self.results = random_data
            self.results_dir = results_dir
            self.figures_dir = figures_dir
            self.test_configs = {
                "packet_rates": packet_rates,
                "packet_loss_rates": [loss/100 for loss in packet_loss_rates],
                "mobility_speeds": mobility_speeds
            }
    
    test = RandomTest()
    
    # Generate figures
    test.generate_packet_rate_figures = types.MethodType(PerformanceTest.generate_packet_rate_figures, test)
    test.generate_packet_loss_figures = types.MethodType(PerformanceTest.generate_packet_loss_figures, test)
    test.generate_mobility_figures = types.MethodType(PerformanceTest.generate_mobility_figures, test)
    test.generate_summary_figures = types.MethodType(PerformanceTest.generate_summary_figures, test)
    test._create_comprehensive_figure = types.MethodType(PerformanceTest._create_comprehensive_figure, test)
    test._create_improvement_figure = types.MethodType(PerformanceTest._create_improvement_figure, test)
    
    test.generate_packet_rate_figures()
    test.generate_packet_loss_figures()
    test.generate_mobility_figures()
    test.generate_summary_figures()
    
    print(f"Random test data and figures generated in {results_dir}")
    return test


# Main function to run the test
if __name__ == "__main__":
    import types
    
    # If no network is provided, generate random test data
    generate_random_test_data()