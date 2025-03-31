import json
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
from typing import Dict, List
import csv
import logging

class OpportunisticSaver:
    """Saver for opportunistic network experiment data"""
    
    def __init__(self, experiment_name=None, crdt_enabled=True):
        # Create logs directory if it doesn't exist
        self.base_dir = "logs"
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        self.exp_dir = os.path.join(self.base_dir, self.experiment_name)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            
        self.crdt_enabled = crdt_enabled
        self.metrics = defaultdict(list)
        self.metrics['hop_count'] = []  # Add hop_count to metrics
        self._start_time = time.time()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=os.path.join(self.exp_dir, 'experiment.log')
        )
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.info(f"Initialized experiment: {self.experiment_name}")
        
        # Create placeholders for metrics
        self.packet_deliveries = []
        self.packet_stats = []
        self.encounter_events = []
        self.throughput_stats = []
    
    def log_encounter(self, node_name: str, neighbor_name: str, rssi: float, position: tuple):
        """Log encounter information"""
        entry = {
            'timestamp': time.time(),
            'node': node_name,
            'neighbor': neighbor_name,
            'rssi': rssi,
            'position': position,
        }
        self.encounter_events.append(entry)
        self.logger.info(f"Logged encounter: {node_name} -> {neighbor_name}")
        
    def log_packet_delivery(self, source: str, destination: str, delay: float, success: bool, metrics: dict = None):
        """Log packet delivery information with hop count and throughput metrics"""
        entry = {
            'timestamp': time.time(),
            'source': source,
            'destination': destination,
            'delay': delay,
            'success': success,
            'crdt_enabled': self.crdt_enabled
        }
        
        # Add hop count and other metrics if provided
        if metrics:
            entry.update(metrics)
            if 'hop_count' in metrics:
                self.metrics['hop_count'].append(metrics['hop_count'])
        
        self.packet_deliveries.append(entry)
        
        # Update metrics
        self.metrics['delays'].append(delay)
        self.metrics['success_rate'].append(1 if success else 0)
        if metrics:
            self.metrics['throughput'].append(metrics.get('throughput', 0))
            self.metrics['bytes_delivered'].append(metrics.get('packet_size', 0))
        
        self.logger.info(
            f"Logged packet delivery: {source} -> {destination}, "
            f"delay: {delay:.2f}s, success: {success}, "
            f"hop count: {metrics.get('hop_count', 'unknown')}, "
            f"throughput: {metrics.get('throughput', 0):.2f} packets/s"
        )
    
    def log_packet_reach_destination(self, source: str, destination: str, delay: float, success: bool):
        """Log packet reach destination information"""
        entry = {
            'timestamp': time.time(),
            'source': source,
            'destination': destination, 
            'delay': delay,
            'success': success,
            'crdt_enabled': self.crdt_enabled
        }
        
        self.packet_deliveries.append(entry)
        
    def log_packet_stats(self, node_name: str, stats: dict):
        """Log packet statistics"""
        entry = {
            'timestamp': time.time(),
            'node': node_name,
            'stats': stats,
            'crdt_enabled': self.crdt_enabled
        }
        self.packet_stats.append(entry)
        
        # Update metrics
        self.metrics['throughput'].append(stats.get('delivered', 0))
        self.metrics['overhead'].append(stats.get('forwarded', 0))
        self.logger.info(f"Logged packet stats for {node_name}: {stats}")
    
    def log_throughput_stats(self, node_name: str, stats: dict):
        """Log periodic throughput statistics"""
        entry = {
            'timestamp': time.time(),
            'node': node_name,
            'stats': stats,
            'crdt_enabled': self.crdt_enabled
        }
        
        self.throughput_stats.append(entry)
        
        # Update aggregate metrics
        self.metrics['node_throughput'][node_name] = stats['current_throughput']
        self.metrics['node_bytes_throughput'][node_name] = stats['bytes_throughput']
        
        self.logger.info(
            f"Logged throughput stats for {node_name}: "
            f"current: {stats['current_throughput']:.2f} packets/s, "
            f"average: {stats['average_throughput']:.2f} packets/s"
        )
    
    def log_packet_expiry(self, source: str, destination: str, packet_id: str, stored_time: float, ttl: float):
        """Log packet expiry due to TTL"""
        entry = {
            'timestamp': time.time(),
            'source': source,
            'destination': destination,
            'packet_id': packet_id,
            'stored_time': stored_time,
            'ttl': ttl,
            'success': False,
            'reason': 'TTL expired',
            'crdt_enabled': self.crdt_enabled
        }
        
        # Record packet expiry
        self.packet_deliveries.append(entry)
        
        # Update metrics to record failure
        self.metrics['success_rate'].append(0)
        
        self.logger.info(
            f"Packet expired: {source} -> {destination}, "
            f"Packet ID: {packet_id}, stored for: {time.time() - stored_time:.2f}s, TTL: {ttl}s"
        )
    
    def generate_performance_graphs(self):
        """Generate performance comparison graphs"""
        figures_dir = os.path.join(self.exp_dir, 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        self.logger.info("Generating performance graphs")
        
        # Skip graph generation if no metrics are available
        if not self.metrics.get('delays'):
            self.logger.warning("No delay metrics available for graphs")
            return
            
        # 1. Delivery Delay Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.metrics['delays'], bins=20, alpha=0.75)
        plt.title('Packet Delivery Delay Distribution')
        plt.xlabel('Delay (seconds)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(figures_dir, 'delay_distribution.png'))
        plt.close()
        
        # Skip remaining graphs if not enough data points
        if len(self.metrics['delays']) < 2:
            self.logger.warning("Not enough data points for remaining graphs")
            return
            
        # 2. Success Rate Over Time
        plt.figure(figsize=(10, 6))
        success_rate = np.cumsum(self.metrics['success_rate']) / np.arange(1, len(self.metrics['success_rate']) + 1)
        plt.plot(success_rate)
        plt.title('Packet Delivery Success Rate Over Time')
        plt.xlabel('Number of Packets')
        plt.ylabel('Success Rate')
        plt.savefig(os.path.join(figures_dir, 'success_rate.png'))
        plt.close()
        
        # 3. Throughput Over Time
        if self.metrics.get('throughput'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['throughput'])
            plt.title('Network Throughput Over Time')
            plt.xlabel('Time')
            plt.ylabel('Packets Delivered')
            plt.savefig(os.path.join(figures_dir, 'throughput.png'))
            plt.close()
        
        # 4. Network Overhead
        if self.metrics.get('overhead'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['overhead'])
            plt.title('Network Overhead Over Time')
            plt.xlabel('Time')
            plt.ylabel('Forwarded Packets')
            plt.savefig(os.path.join(figures_dir, 'overhead.png'))
            plt.close()
        
        # 5. Hop Count Distribution
        if self.metrics.get('hop_count'):
            plt.figure(figsize=(10, 6))
            plt.hist(self.metrics['hop_count'], bins=min(10, max(self.metrics['hop_count'])+1), alpha=0.75)
            plt.title('Hop Count Distribution')
            plt.xlabel('Number of Hops')
            plt.ylabel('Frequency')
            plt.xticks(range(0, max(self.metrics['hop_count'])+1))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(figures_dir, 'hop_count_distribution.png'))
            plt.close()
            
            # 6. Hop Count vs Delay Scatter Plot
            if len(self.metrics['hop_count']) == len(self.metrics['delays']):
                plt.figure(figsize=(10, 6))
                plt.scatter(self.metrics['hop_count'], self.metrics['delays'], alpha=0.6)
                plt.title('Hop Count vs Delivery Delay')
                plt.xlabel('Number of Hops')
                plt.ylabel('Delay (seconds)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(figures_dir, 'hop_count_vs_delay.png'))
                plt.close()
        
        # Save metrics summary
        summary = self._calculate_aggregate_stats()
        with open(os.path.join(self.exp_dir, 'performance_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info("Completed performance graph generation")

    def generate_throughput_graphs(self):
        """Generate throughput-specific graphs"""
        figures_dir = os.path.join(self.exp_dir, 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        # 1. Throughput over time
        plt.figure(figsize=(12, 6))
        timestamps = [entry['timestamp'] - self._start_time for entry in self.throughput_stats]
        throughputs = [entry['stats']['current_throughput'] for entry in self.throughput_stats]
        
        plt.plot(timestamps, throughputs, 'b-', linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title('Network Throughput Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Throughput (packets/s)')
        plt.savefig(os.path.join(figures_dir, 'throughput_time_series.png'), dpi=300)
        plt.close()
        
        # 2. Per-node throughput comparison
        if self.metrics['node_throughput']:
            plt.figure(figsize=(10, 6))
            nodes = list(self.metrics['node_throughput'].keys())
            throughputs = [self.metrics['node_throughput'][node] for node in nodes]
            
            plt.bar(nodes, throughputs)
            plt.title('Average Throughput by Node')
            plt.xlabel('Node')
            plt.ylabel('Throughput (packets/s)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'node_throughput_comparison.png'), dpi=300)
            plt.close()

    def save_experiment_results(self, experiment_config: Dict):
        """Save all experiment results after completion"""
        results_dir = os.path.join(self.exp_dir, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        self.logger.info(f"Saving experiment results to {results_dir}")
        
        try:
            # 1. Save experiment configuration
            config_file = os.path.join(results_dir, 'experiment_config.json')
            with open(config_file, 'w') as f:
                json.dump(experiment_config, f, indent=2)
                
            # 2. Save packet deliveries to CSV
            if self.packet_deliveries:
                deliveries_file = os.path.join(results_dir, 'packet_deliveries.csv')
                df = pd.DataFrame(self.packet_deliveries)
                df.to_csv(deliveries_file, index=False)
                self.logger.info(f"Saved {len(self.packet_deliveries)} packet deliveries")
            else:
                self.logger.warning("No packet deliveries to save")
                
            # 3. Save packet stats to CSV
            if self.packet_stats:
                # Extract the nested stats dictionary
                flattened_stats = []
                for entry in self.packet_stats:
                    record = {
                        'timestamp': entry['timestamp'],
                        'node': entry['node'],
                        'crdt_enabled': entry['crdt_enabled']
                    }
                    # Add the stats values directly to the record
                    for k, v in entry['stats'].items():
                        record[k] = v
                    flattened_stats.append(record)
                
                stats_file = os.path.join(results_dir, 'packet_stats.csv')
                df = pd.DataFrame(flattened_stats)
                df.to_csv(stats_file, index=False)
                self.logger.info(f"Saved {len(self.packet_stats)} packet stats entries")
            else:
                self.logger.warning("No packet stats to save")
                
            # 4. Save encounters to CSV
            if self.encounter_events:
                encounters_file = os.path.join(results_dir, 'encounters.csv')
                df = pd.DataFrame(self.encounter_events)
                df.to_csv(encounters_file, index=False)
                self.logger.info(f"Saved {len(self.encounter_events)} encounter events")
            else:
                self.logger.warning("No encounter events to save")
                
            # 5. Save aggregate statistics
            stats = self._calculate_aggregate_stats()
            stats_file = os.path.join(results_dir, 'aggregate_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
            self.logger.info("Successfully saved experiment results")
            return results_dir
            
        except Exception as e:
            self.logger.error(f"Error saving experiment results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _calculate_aggregate_stats(self) -> Dict:
        """Calculate aggregate statistics from metrics"""
        stats = {
            'experiment_duration': time.time() - self._start_time,
            'crdt_enabled': self.crdt_enabled
        }
        
        # Only calculate statistics if we have data
        if self.metrics.get('delays'):
            stats.update({
                'total_packets': len(self.metrics['delays']),
                'average_delay': float(np.mean(self.metrics['delays'])),
                'delay_std': float(np.std(self.metrics['delays'])) if len(self.metrics['delays']) > 1 else 0,
                'success_rate': float(np.mean(self.metrics['success_rate'])) if self.metrics.get('success_rate') else 0,
                'total_throughput': int(sum(self.metrics['throughput'])) if self.metrics.get('throughput') else 0,
                'average_overhead': float(np.mean(self.metrics['overhead'])) if self.metrics.get('overhead') else 0
            })
            
            # Add hop count statistics if available
            if self.metrics.get('hop_count'):
                stats.update({
                    'average_hop_count': float(np.mean(self.metrics['hop_count'])),
                    'max_hop_count': int(max(self.metrics['hop_count'])),
                    'min_hop_count': int(min(self.metrics['hop_count'])),
                    'hop_count_std': float(np.std(self.metrics['hop_count'])) if len(self.metrics['hop_count']) > 1 else 0
                })
        else:
            stats.update({
                'total_packets': 0,
                'average_delay': 0,
                'delay_std': 0,
                'success_rate': 0,
                'total_throughput': 0,
                'average_overhead': 0,
                'average_hop_count': 0,
                'max_hop_count': 0,
                'min_hop_count': 0,
                'hop_count_std': 0
            })
        
        return stats
