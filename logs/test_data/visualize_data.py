#!/usr/bin/env python3

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set the style for the plots
plt.style.use('ggplot')
sns.set_palette("Set2")
sns.set_context("paper", font_scale=1.2)

# Constants
DATA_DIR = "logs/test_data"
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
RESULTS_FILE = os.path.join(DATA_DIR, "performance_results.json")
CSV_FILE = os.path.join(DATA_DIR, "performance_results.csv")

# Create figures directory
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

def load_data():
    """Load the generated performance data"""
    # Try to load CSV first (faster and simpler for aggregate metrics)
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    
    # Fall back to JSON
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    raise FileNotFoundError(f"No data found in {DATA_DIR}")

def calculate_averages(df):
    """Calculate averages across runs for each node count and CRDT setting"""
    # Group by node count and CRDT usage, then average across runs
    return df.groupby(['node_count', 'use_crdt']).mean().reset_index()

def plot_latency_comparison(df):
    """Plot latency comparison between CRDT and non-CRDT approaches"""
    plt.figure(figsize=(10, 6))
    
    avg_df = calculate_averages(df)
    
    # Plot average latency vs. node count
    sns.lineplot(
        data=avg_df, 
        x='node_count', 
        y='average_latency', 
        hue='use_crdt',
        style='use_crdt',
        markers=True,
        dashes=False,
        err_style='band'
    )
    
    plt.title('Average Latency vs. Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Latency (seconds)')
    plt.xticks(sorted(df['node_count'].unique()))
    plt.legend(labels=['Without CRDT', 'With CRDT'])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'latency_comparison.png'), dpi=300)
    print(f"Saved latency comparison plot to {FIGURES_DIR}")

def plot_throughput_comparison(df):
    """Plot throughput comparison between CRDT and non-CRDT approaches"""
    plt.figure(figsize=(10, 6))
    
    avg_df = calculate_averages(df)
    
    # Plot average throughput vs. node count
    sns.lineplot(
        data=avg_df, 
        x='node_count', 
        y='average_throughput', 
        hue='use_crdt',
        style='use_crdt',
        markers=True,
        dashes=False,
        err_style='band'
    )
    
    plt.title('Average Throughput vs. Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Throughput (packets/second)')
    plt.xticks(sorted(df['node_count'].unique()))
    plt.legend(labels=['Without CRDT', 'With CRDT'])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'throughput_comparison.png'), dpi=300)
    print(f"Saved throughput comparison plot to {FIGURES_DIR}")

def plot_success_rate_comparison(df):
    """Plot success rate comparison between CRDT and non-CRDT approaches"""
    plt.figure(figsize=(10, 6))
    
    avg_df = calculate_averages(df)
    
    # Plot success rate vs. node count
    sns.lineplot(
        data=avg_df, 
        x='node_count', 
        y='success_rate', 
        hue='use_crdt',
        style='use_crdt',
        markers=True,
        dashes=False,
        err_style='band'
    )
    
    plt.title('Packet Delivery Success Rate vs. Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Success Rate')
    plt.xticks(sorted(df['node_count'].unique()))
    plt.legend(labels=['Without CRDT', 'With CRDT'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'success_rate_comparison.png'), dpi=300)
    print(f"Saved success rate comparison plot to {FIGURES_DIR}")

def plot_missing_packets_comparison(df):
    """Plot missing packets comparison between CRDT and non-CRDT approaches"""
    plt.figure(figsize=(10, 6))
    
    avg_df = calculate_averages(df)
    
    # Calculate missing packet percentage
    avg_df['missing_packet_pct'] = avg_df['missing_packets'] / avg_df['packet_count']
    
    # Plot missing packet percentage vs. node count
    sns.lineplot(
        data=avg_df, 
        x='node_count', 
        y='missing_packet_pct', 
        hue='use_crdt',
        style='use_crdt',
        markers=True,
        dashes=False,
        err_style='band'
    )
    
    plt.title('Missing Packets Percentage vs. Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Missing Packets (%)')
    plt.xticks(sorted(df['node_count'].unique()))
    plt.legend(labels=['Without CRDT', 'With CRDT'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'missing_packets_comparison.png'), dpi=300)
    print(f"Saved missing packets comparison plot to {FIGURES_DIR}")

def plot_hop_count_comparison(df):
    """Plot hop count comparison between CRDT and non-CRDT approaches"""
    plt.figure(figsize=(10, 6))
    
    avg_df = calculate_averages(df)
    
    # Plot average hop count vs. node count
    sns.lineplot(
        data=avg_df, 
        x='node_count', 
        y='average_hop_count', 
        hue='use_crdt',
        style='use_crdt',
        markers=True,
        dashes=False,
        err_style='band'
    )
    
    plt.title('Average Hop Count vs. Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Hop Count')
    plt.xticks(sorted(df['node_count'].unique()))
    plt.legend(labels=['Without CRDT', 'With CRDT'])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hop_count_comparison.png'), dpi=300)
    print(f"Saved hop count comparison plot to {FIGURES_DIR}")

def plot_all_metrics_radar(df):
    """Create a radar chart comparing all metrics for CRDT vs non-CRDT"""
    # For radar chart, we'll use the 10-node configuration as representative
    radar_df = df[df['node_count'] == 10].copy()
    avg_radar_df = calculate_averages(radar_df)
    
    # Convert metrics to relative performance (0-1 scale)
    metrics = ['average_latency', 'average_throughput', 'success_rate', 'average_hop_count']
    normalized_df = pd.DataFrame()
    
    for metric in metrics:
        # For latency and hop count, lower is better, so invert the normalization
        if metric in ['average_latency', 'average_hop_count']:
            max_val = avg_radar_df[metric].max()
            min_val = avg_radar_df[metric].min()
            if max_val != min_val:  # Avoid division by zero
                avg_radar_df[f'{metric}_norm'] = 1 - ((avg_radar_df[metric] - min_val) / (max_val - min_val))
            else:
                avg_radar_df[f'{metric}_norm'] = 1.0
        else:
            # For throughput and success rate, higher is better
            max_val = avg_radar_df[metric].max()
            min_val = avg_radar_df[metric].min()
            if max_val != min_val:  # Avoid division by zero
                avg_radar_df[f'{metric}_norm'] = (avg_radar_df[metric] - min_val) / (max_val - min_val)
            else:
                avg_radar_df[f'{metric}_norm'] = 1.0
    
    # Prepare radar chart
    plt.figure(figsize=(10, 8))
    
    # Radar chart setup
    categories = ['Latency\n(lower is better)', 'Throughput\n(higher is better)', 
                  'Success Rate\n(higher is better)', 'Hop Count\n(lower is better)']
    N = len(categories)
    
    # Create angle for each metric (evenly spaced around the circle)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Get normalized values for each approach
    values_no_crdt = avg_radar_df[avg_radar_df['use_crdt'] == False][[
        'average_latency_norm', 'average_throughput_norm', 
        'success_rate_norm', 'average_hop_count_norm'
    ]].values.flatten().tolist()
    values_no_crdt += values_no_crdt[:1]  # Close the loop
    
    values_crdt = avg_radar_df[avg_radar_df['use_crdt'] == True][[
        'average_latency_norm', 'average_throughput_norm', 
        'success_rate_norm', 'average_hop_count_norm'
    ]].values.flatten().tolist()
    values_crdt += values_crdt[:1]  # Close the loop
    
    # Create the radar plot
    ax = plt.subplot(111, polar=True)
    
    # Draw the polygon and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Plot non-CRDT values
    ax.plot(angles, values_no_crdt, 'b-', linewidth=2, label='Without CRDT')
    ax.fill(angles, values_no_crdt, 'blue', alpha=0.1)
    
    # Plot CRDT values
    ax.plot(angles, values_crdt, 'r-', linewidth=2, label='With CRDT')
    ax.fill(angles, values_crdt, 'red', alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Performance Comparison: CRDT vs. Non-CRDT (10 Nodes)', size=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'radar_comparison.png'), dpi=300)
    print(f"Saved radar comparison plot to {FIGURES_DIR}")

def plot_improvement_percentage(df):
    """Plot the percentage improvement from using CRDT across node counts"""
    plt.figure(figsize=(12, 8))
    
    # Group data by node count and CRDT usage
    grouped = df.groupby(['node_count', 'use_crdt']).mean().reset_index()
    
    # Pivot to have separate columns for CRDT and non-CRDT
    pivot_latency = grouped.pivot(index='node_count', columns='use_crdt', values='average_latency')
    pivot_throughput = grouped.pivot(index='node_count', columns='use_crdt', values='average_throughput')
    pivot_success = grouped.pivot(index='node_count', columns='use_crdt', values='success_rate')
    pivot_missing = grouped.pivot(index='node_count', columns='use_crdt', values='missing_packets')
    
    # Calculate improvement percentages
    improvements = pd.DataFrame(index=pivot_latency.index)
    
    # For latency, lower is better, so calculate how much CRDT increases latency
    improvements['latency'] = ((pivot_latency[True] - pivot_latency[False]) / pivot_latency[False]) * 100
    
    # For throughput, higher is better
    improvements['throughput'] = ((pivot_throughput[True] - pivot_throughput[False]) / pivot_throughput[False]) * 100
    
    # For success rate, higher is better
    improvements['success_rate'] = ((pivot_success[True] - pivot_success[False]) / pivot_success[False]) * 100
    
    # For missing packets, lower is better, so calculate how much CRDT reduces missing packets
    improvements['missing_packets'] = -((pivot_missing[True] - pivot_missing[False]) / pivot_missing[False]) * 100
    
    # Plot improvement percentages
    ax = improvements.plot(kind='bar', figsize=(12, 7), width=0.8)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.title('Percentage Improvement from Using CRDT by Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Improvement (%)')
    plt.xticks(rotation=0)
    plt.legend(['Latency (negative is better)', 
                'Throughput (positive is better)',
                'Success Rate (positive is better)', 
                'Missing Packets Reduction (positive is better)'])
    
    # Add value labels on the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 10 if p.get_height() > 0 else -10), 
                    textcoords='offset points',
                    fontsize=8)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'improvement_percentage.png'), dpi=300)
    print(f"Saved improvement percentage plot to {FIGURES_DIR}")

def create_summary_table(df):
    """Create and save a summary table of all metrics"""
    # Group data by node count and CRDT usage
    summary = df.groupby(['node_count', 'use_crdt']).agg({
        'average_latency': 'mean',
        'average_throughput': 'mean',
        'success_rate': 'mean',
        'missing_packets': 'mean',
        'average_hop_count': 'mean'
    }).reset_index()
    
    # Format the summary table
    summary = summary.round(3)
    
    # Save to CSV
    summary.to_csv(os.path.join(FIGURES_DIR, 'performance_summary.csv'), index=False)
    
    # Also create an HTML version for easy viewing
    html = summary.to_html(index=False)
    with open(os.path.join(FIGURES_DIR, 'performance_summary.html'), 'w') as f:
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
                .true {
                    background-color: #e6f7ff;
                }
                .false {
                    background-color: #fff2e6;
                }
            </style>
        </head>
        <body>
            <h2>Performance Metrics Summary: CRDT vs. Non-CRDT</h2>
        """ + html + """
        </body>
        </html>
        """)
    
    print(f"Saved performance summary table to {FIGURES_DIR}")

def main():
    """Main function to generate all visualizations"""
    # Load data
    df = load_data()
    
    # Generate plots
    plot_latency_comparison(df)
    plot_throughput_comparison(df)
    plot_success_rate_comparison(df)
    plot_missing_packets_comparison(df)
    plot_hop_count_comparison(df)
    plot_all_metrics_radar(df)
    plot_improvement_percentage(df)
    
    # Create summary table
    create_summary_table(df)
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main() 