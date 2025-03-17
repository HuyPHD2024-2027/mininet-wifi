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
        
    def log_packet_delivery(self, source: str, destination: str, delay: float, success: bool):
        """Log packet delivery information"""
        entry = {
            'timestamp': time.time(),
            'source': source,
            'destination': destination,
            'delay': delay,
            'success': success,
            'crdt_enabled': self.crdt_enabled
        }
        self.packet_deliveries.append(entry)
        
        # Update metrics
        self.metrics['delays'].append(delay)
        self.metrics['success_rate'].append(1 if success else 0)
        self.logger.info(f"Logged packet delivery: {source} -> {destination}, delay: {delay:.2f}s, success: {success}")
    
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
        
        # Save metrics summary
        summary = self._calculate_aggregate_stats()
        with open(os.path.join(self.exp_dir, 'performance_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info("Completed performance graph generation")

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
        else:
            stats.update({
                'total_packets': 0,
                'average_delay': 0,
                'delay_std': 0,
                'success_rate': 0,
                'total_throughput': 0,
                'average_overhead': 0
            })
        
        return stats
