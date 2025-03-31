#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set the style for the plots
plt.style.use('ggplot')
sns.set_palette("Set2")
sns.set_context("paper", font_scale=1.2)

# Paths
REAL_DATA_DIR = "../experiment_5nodes_120s"
TEST_DATA_DIR = "."
FIGURES_DIR = os.path.join(TEST_DATA_DIR, "real_data_figures")

# Ensure figures directory exists
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

def load_real_data():
    """Load the real experiment data from experiment_5nodes_120s"""
    # Read performance summary
    performance_summary_path = os.path.join(REAL_DATA_DIR, "performance_summary.json")
    
    if not os.path.exists(performance_summary_path):
        print(f"Error: Could not find real data at {performance_summary_path}")
        return None
    
    with open(performance_summary_path, 'r') as f:
        summary = json.load(f)
    
    # Check if packet deliveries data is available
    results_dir = os.path.join(REAL_DATA_DIR, "results")
    packet_deliveries_path = os.path.join(results_dir, "packet_deliveries.csv")
    
    if os.path.exists(packet_deliveries_path):
        deliveries_df = pd.read_csv(packet_deliveries_path)
        return {"summary": summary, "deliveries": deliveries_df}
    else:
        return {"summary": summary}

def analyze_hop_counts(real_data):
    """Analyze hop count distribution from real data"""
    if "deliveries" not in real_data:
        print("No delivery data available for hop count analysis")
        return
    
    deliveries_df = real_data["deliveries"]
    
    # Check if hop_count column exists
    if "hop_count" not in deliveries_df.columns:
        print("No hop count data available in deliveries")
        return
    
    # Filter only successful deliveries
    successful_deliveries = deliveries_df[deliveries_df["success"] == True]
    
    # Plot hop count distribution
    plt.figure(figsize=(10, 6))
    
    # Get hop counts
    hop_counts = successful_deliveries["hop_count"].dropna()
    
    if len(hop_counts) == 0:
        print("No valid hop count data found")
        return
    
    # Plot histogram
    sns.histplot(hop_counts, bins=range(int(hop_counts.min()), int(hop_counts.max()) + 2),
                 kde=False, stat="count", color="blue", alpha=0.7)
    
    plt.title("Hop Count Distribution (Real Data)")
    plt.xlabel("Number of Hops")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "real_hop_count_distribution.png"), dpi=300)
    print(f"Saved real hop count distribution to {FIGURES_DIR}")
    
    # Hop count vs delay scatter plot
    plt.figure(figsize=(10, 6))
    
    # Check if delay column exists
    if "delay" in successful_deliveries.columns:
        plt.scatter(successful_deliveries["hop_count"], successful_deliveries["delay"], 
                    alpha=0.6, color="blue")
        
        plt.title("Hop Count vs. Delivery Delay (Real Data)")
        plt.xlabel("Number of Hops")
        plt.ylabel("Delay (seconds)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "real_hop_count_vs_delay.png"), dpi=300)
        print(f"Saved real hop count vs delay plot to {FIGURES_DIR}")

def analyze_success_rate(real_data):
    """Analyze packet delivery success rate from real data"""
    summary = real_data["summary"]
    
    # Check if success_rate is available
    if "success_rate" not in summary:
        print("No success rate data available in summary")
        return
    
    # Create visualization of success rate
    plt.figure(figsize=(8, 6))
    
    # Create bar chart of success rate
    plt.bar(["5 Nodes"], [summary["success_rate"]], color="green", alpha=0.7)
    
    plt.title("Packet Delivery Success Rate (Real Data)")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value label
    plt.text(0, summary["success_rate"] + 0.02, f"{summary['success_rate']:.2%}", 
             ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "real_success_rate.png"), dpi=300)
    print(f"Saved real success rate plot to {FIGURES_DIR}")

def analyze_latency(real_data):
    """Analyze packet delivery latency from real data"""
    if "deliveries" not in real_data:
        print("No delivery data available for latency analysis")
        return
    
    deliveries_df = real_data["deliveries"]
    
    # Check if delay column exists
    if "delay" not in deliveries_df.columns:
        print("No delay data available in deliveries")
        return
    
    # Filter only successful deliveries
    successful_deliveries = deliveries_df[deliveries_df["success"] == True]
    
    # Plot latency distribution
    plt.figure(figsize=(10, 6))
    
    # Get delays
    delays = successful_deliveries["delay"].dropna()
    
    if len(delays) == 0:
        print("No valid delay data found")
        return
    
    # Plot histogram
    sns.histplot(delays, bins=20, kde=True, color="purple", alpha=0.7)
    
    plt.title("Packet Delivery Delay Distribution (Real Data)")
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add vertical line for average delay
    avg_delay = delays.mean()
    plt.axvline(x=avg_delay, color='red', linestyle='--', 
                label=f"Avg: {avg_delay:.2f}s")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "real_delay_distribution.png"), dpi=300)
    print(f"Saved real delay distribution to {FIGURES_DIR}")

def compare_with_simulated_data():
    """Compare real data with simulated data"""
    # Load simulated data
    simulated_csv = os.path.join(TEST_DATA_DIR, "performance_results.csv")
    
    if not os.path.exists(simulated_csv):
        print(f"Simulated data not found at {simulated_csv}")
        print("Run generate_data.py first")
        return
    
    simulated_df = pd.read_csv(simulated_csv)
    
    # Filter for 5 nodes only (to match real data)
    sim_5nodes_df = simulated_df[simulated_df["node_count"] == 5].copy()
    
    # Group by CRDT usage and average across runs
    sim_5nodes_avg = sim_5nodes_df.groupby('use_crdt').mean().reset_index()
    
    # Load real data summary
    real_data = load_real_data()
    if not real_data:
        return
    
    real_summary = real_data["summary"]
    
    # Create comparison charts
    metrics = [
        {"name": "success_rate", "title": "Success Rate", "real_value": real_summary.get("success_rate", 0), 
         "format": "percentage", "higher_better": True},
        {"name": "average_latency", "title": "Latency (s)", "real_value": real_summary.get("average_delay", 0), 
         "format": "float", "higher_better": False},
        {"name": "average_hop_count", "title": "Avg Hop Count", "real_value": real_summary.get("average_hop_count", 0), 
         "format": "float", "higher_better": False}
    ]
    
    # Create a comparison figure with all metrics
    plt.figure(figsize=(12, 8))
    
    # Set up bar positions
    bar_width = 0.2
    index = np.arange(len(metrics))
    
    # Plot bars for each data source
    sim_no_crdt = plt.bar(index - bar_width, 
                           [sim_5nodes_avg[sim_5nodes_avg['use_crdt'] == False][m['name']].values[0] for m in metrics], 
                           bar_width, label='Simulated (No CRDT)', color='lightblue')
    
    sim_crdt = plt.bar(index, 
                       [sim_5nodes_avg[sim_5nodes_avg['use_crdt'] == True][m['name']].values[0] for m in metrics], 
                       bar_width, label='Simulated (CRDT)', color='darkblue')
    
    real_data_bars = plt.bar(index + bar_width, 
                            [m['real_value'] for m in metrics], 
                            bar_width, label='Real Data', color='green')
    
    # Add labels, title and legend
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Comparison: Real Data vs. Simulated Data (5 Nodes)')
    plt.xticks(index, [m['title'] for m in metrics])
    plt.legend()
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f"{height:.2f}", ha='center', va='bottom', fontsize=9)
    
    add_labels(sim_no_crdt)
    add_labels(sim_crdt)
    add_labels(real_data_bars)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "real_vs_simulated.png"), dpi=300)
    print(f"Saved real vs simulated comparison to {FIGURES_DIR}")
    
    # Create a summary table
    comparison_data = {
        "Metric": [m['title'] for m in metrics],
        "Real Data": [m['real_value'] for m in metrics],
        "Simulated (No CRDT)": [sim_5nodes_avg[sim_5nodes_avg['use_crdt'] == False][m['name']].values[0] for m in metrics],
        "Simulated (CRDT)": [sim_5nodes_avg[sim_5nodes_avg['use_crdt'] == True][m['name']].values[0] for m in metrics]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(FIGURES_DIR, "real_vs_simulated_comparison.csv"), index=False)
    
    # Also create HTML version
    html = comparison_df.to_html(index=False)
    with open(os.path.join(FIGURES_DIR, "real_vs_simulated_comparison.html"), 'w') as f:
        f.write("""
        <html>
        <head>
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                }
                th, td {
                    text-align: right;
                    padding: 8px;
                    border: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                td:first-child, th:first-child {
                    text-align: left;
                }
            </style>
        </head>
        <body>
            <h2>Real Data vs. Simulated Data Comparison (5 Nodes)</h2>
        """ + html + """
        </body>
        </html>
        """)

def main():
    """Main function to analyze real data and compare with simulated data"""
    print(f"Loading real data from {REAL_DATA_DIR}...")
    real_data = load_real_data()
    
    if not real_data:
        print("Failed to load real data. Please check the paths.")
        return
    
    print("Analyzing real data...")
    analyze_hop_counts(real_data)
    analyze_success_rate(real_data)
    analyze_latency(real_data)
    
    # Compare with simulated data
    try:
        print("Comparing with simulated data...")
        compare_with_simulated_data()
    except Exception as e:
        print(f"Error comparing with simulated data: {e}")
    
    print(f"Analysis complete. Results saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main() 