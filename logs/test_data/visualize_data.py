#!/usr/bin/env python3

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy import interpolate

# Set the style for the plots
plt.style.use('ggplot')
sns.set_palette("Set2")
sns.set_context("paper", font_scale=1.3)  # Slightly larger font for better readability

# Constants
DATA_DIR = "logs/test_data"
FIGURES_DIR = "figures"
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

def create_smooth_line(x, y, num_points=100):
    """Create a smooth curve using spline interpolation"""
    # Sort x and y based on x values to ensure interpolation works correctly
    sorted_indices = np.argsort(x)
    x_sorted = np.array(x)[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]
    
    # Create the interpolation function
    spline = interpolate.make_interp_spline(x_sorted, y_sorted, k=3)
    
    # Generate smoothed x and y values
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
    y_smooth = spline(x_smooth)
    
    return x_smooth, y_smooth

def plot_latency_comparison(df):
    """Plot latency comparison between CRDT and non-CRDT approaches with smooth curves"""
    plt.figure(figsize=(12, 7))
    
    avg_df = calculate_averages(df)
    
    # Create custom palette with boolean keys
    palette = {True: 'green', False: 'red'}
    
    # Store values at 50 nodes for annotation
    values_at_50_nodes = {}
    
    # For smoother lines, plot custom curves instead of using seaborn
    for use_crdt in [True, False]:
        subset = avg_df[avg_df['use_crdt'] == use_crdt]
        x = subset['node_count'].values
        y = subset['average_latency'].values
        
        # Store the value at 50 nodes (or the maximum node count)
        max_node_idx = np.argmax(x)
        values_at_50_nodes[use_crdt] = y[max_node_idx]
        
        # Create smooth curves
        x_smooth, y_smooth = create_smooth_line(x, y)
        
        color = palette[use_crdt]
        marker = 'o' if use_crdt else 's'
        label = 'CRDT' if use_crdt else 'Non-CRDT'
        
        # Plot the smooth line
        plt.plot(x_smooth, y_smooth, color=color, linewidth=3, label=label)
        
        # Add markers for the actual data points
        plt.scatter(x, y, color=color, marker=marker, s=100, zorder=5)
    
    plt.title('Packet Delay vs. Number of Nodes', fontsize=16)
    plt.xlabel('Number of Nodes', fontsize=14)
    plt.ylabel('Average Delay (seconds)', fontsize=14)
    plt.xticks(sorted(df['node_count'].unique()), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis range and ensure x-axis starts at 5
    plt.ylim(2, 15)  # Updated to match new data range
    plt.xlim(4, None)  # Start slightly before 5 for better visualization
    
    # Add legend with custom labels
    plt.legend(title='CRDT Enabled', fontsize=12, title_fontsize=12)
    
    # Add annotations for values at 50 nodes
    crdt_val = values_at_50_nodes[True]
    non_crdt_val = values_at_50_nodes[False]
    
    plt.annotate(f'CRDT: {crdt_val:.1f}s', 
                xy=(50, crdt_val), 
                xytext=(50, crdt_val+0.5),
                fontsize=12, 
                color='green',
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8))
    
    plt.annotate(f'Non-CRDT: {non_crdt_val:.1f}s', 
                xy=(50, non_crdt_val), 
                xytext=(50, non_crdt_val+0.8),
                fontsize=12, 
                color='red',
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'delay_comparison.png'), dpi=300)
    print(f"Saved delay comparison plot to {FIGURES_DIR}")

def plot_hop_count_comparison(df):
    """Plot hop count comparison between CRDT and non-CRDT approaches with smooth curves"""
    plt.figure(figsize=(12, 7))
    
    avg_df = calculate_averages(df)
    
    # Create custom palette with boolean keys
    palette = {True: 'green', False: 'red'}
    
    # Store values at 50 nodes for annotation
    values_at_50_nodes = {}
    
    # For smoother lines, plot custom curves instead of using seaborn
    for use_crdt in [True, False]:
        subset = avg_df[avg_df['use_crdt'] == use_crdt]
        x = subset['node_count'].values
        y = subset['average_hop_count'].values
        
        # Store the value at 50 nodes (or the maximum node count)
        max_node_idx = np.argmax(x)
        values_at_50_nodes[use_crdt] = y[max_node_idx]
        
        # Create smooth curves
        x_smooth, y_smooth = create_smooth_line(x, y)
        
        color = palette[use_crdt]
        marker = 'o' if use_crdt else 's'
        label = 'CRDT' if use_crdt else 'Non-CRDT'
        
        # Plot the smooth line
        plt.plot(x_smooth, y_smooth, color=color, linewidth=3, label=label)
        
        # Add markers for the actual data points
        plt.scatter(x, y, color=color, marker=marker, s=100, zorder=5)
    
    plt.title('Average Hop Count vs. Number of Nodes', fontsize=16)
    plt.xlabel('Number of Nodes', fontsize=14)
    plt.ylabel('Average Hop Count', fontsize=14)
    plt.xticks(sorted(df['node_count'].unique()), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis range and ensure x-axis starts at 5
    plt.ylim(0, 15)  # Range adjusted for new data
    plt.xlim(4, None)  # Start slightly before 5 for better visualization
    
    # Add legend with custom labels
    plt.legend(title='CRDT Enabled', fontsize=12, title_fontsize=12)
    
    # Add annotations for values at 50 nodes
    crdt_val = values_at_50_nodes[True]
    non_crdt_val = values_at_50_nodes[False]
    
    plt.annotate(f'CRDT: {crdt_val:.1f}', 
                xy=(50, crdt_val), 
                xytext=(50, crdt_val+0.8),
                fontsize=12, 
                color='green',
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8))
    
    plt.annotate(f'Non-CRDT: {non_crdt_val:.1f}', 
                xy=(50, non_crdt_val), 
                xytext=(50, non_crdt_val+0.8),
                fontsize=12, 
                color='red',
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hop_count_comparison.png'), dpi=300)
    print(f"Saved hop count comparison plot to {FIGURES_DIR}")

def plot_success_rate_comparison(df):
    """Plot success rate comparison between CRDT and non-CRDT approaches with smooth curves"""
    plt.figure(figsize=(12, 7))
    
    avg_df = calculate_averages(df)
    
    # Create custom palette with boolean keys
    palette = {True: 'green', False: 'red'}
    
    # Store values at 50 nodes for annotation
    values_at_50_nodes = {}
    
    # For smoother lines, plot custom curves instead of using seaborn
    for use_crdt in [True, False]:
        subset = avg_df[avg_df['use_crdt'] == use_crdt]
        x = subset['node_count'].values
        y = subset['success_rate'].values * 100  # Convert to percentage
        
        # Store the value at 50 nodes (or the maximum node count)
        max_node_idx = np.argmax(x)
        values_at_50_nodes[use_crdt] = y[max_node_idx]
        
        # Create smooth curves
        x_smooth, y_smooth = create_smooth_line(x, y)
        
        color = palette[use_crdt]
        marker = 'o' if use_crdt else 's'
        label = 'CRDT' if use_crdt else 'Non-CRDT'
        
        # Plot the smooth line
        plt.plot(x_smooth, y_smooth, color=color, linewidth=3, label=label)
        
        # Add markers for the actual data points
        plt.scatter(x, y, color=color, marker=marker, s=100, zorder=5)
    
    plt.title('Packet Delivery Success Rate vs. Number of Nodes', fontsize=16)
    plt.xlabel('Number of Nodes', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.xticks(sorted(df['node_count'].unique()), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis range and ensure x-axis starts at 5
    plt.ylim(75, 100)
    plt.xlim(4, None)  # Start slightly before 5 for better visualization
    
    # Add legend with custom labels
    plt.legend(title='CRDT Enabled', fontsize=12, title_fontsize=12)
    
    # Add annotations for values at 50 nodes
    crdt_val = values_at_50_nodes[True]
    non_crdt_val = values_at_50_nodes[False]
    
    plt.annotate(f'CRDT: {crdt_val:.1f}%', 
                xy=(50, crdt_val), 
                xytext=(50, crdt_val+1),
                fontsize=12, 
                color='green',
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8))
    
    plt.annotate(f'Non-CRDT: {non_crdt_val:.1f}%', 
                xy=(50, non_crdt_val), 
                xytext=(50, non_crdt_val-3),
                fontsize=12, 
                color='red',
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'success_rate_comparison.png'), dpi=300)
    print(f"Saved success rate comparison plot to {FIGURES_DIR}")

def plot_throughput_comparison(df):
    """Plot throughput comparison between CRDT and non-CRDT approaches with smooth curves"""
    plt.figure(figsize=(12, 7))
    
    avg_df = calculate_averages(df)
    
    # Create custom palette with boolean keys
    palette = {True: 'green', False: 'red'}
    
    # Store values at 50 nodes for annotation
    values_at_50_nodes = {}
    
    # For smoother lines, plot custom curves instead of using seaborn
    for use_crdt in [True, False]:
        subset = avg_df[avg_df['use_crdt'] == use_crdt]
        x = subset['node_count'].values
        y = subset['average_throughput'].values
        
        # Store the value at 50 nodes (or the maximum node count)
        max_node_idx = np.argmax(x)
        values_at_50_nodes[use_crdt] = y[max_node_idx]
        
        # Create smooth curves
        x_smooth, y_smooth = create_smooth_line(x, y)
        
        color = palette[use_crdt]
        marker = 'o' if use_crdt else 's'
        label = 'CRDT' if use_crdt else 'Non-CRDT'
        
        # Plot the smooth line
        plt.plot(x_smooth, y_smooth, color=color, linewidth=3, label=label)
        
        # Add markers for the actual data points
        plt.scatter(x, y, color=color, marker=marker, s=100, zorder=5)
    
    plt.title('Average Throughput vs. Number of Nodes', fontsize=16)
    plt.xlabel('Number of Nodes', fontsize=14)
    plt.ylabel('Throughput (packets/second)', fontsize=14)
    plt.xticks(sorted(df['node_count'].unique()), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis range and ensure x-axis starts at 5
    plt.ylim(0, 5)  # Adjusted for new data range
    plt.xlim(4, None)  # Start slightly before 5 for better visualization
    
    # Add legend with custom labels
    plt.legend(title='CRDT Enabled', fontsize=12, title_fontsize=12)
    
    # Add annotations for values at 50 nodes
    crdt_val = values_at_50_nodes[True]
    non_crdt_val = values_at_50_nodes[False]
    
    plt.annotate(f'CRDT: {crdt_val:.2f}', 
                xy=(50, crdt_val), 
                xytext=(50, crdt_val+0.3),
                fontsize=12, 
                color='green',
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8))
    
    plt.annotate(f'Non-CRDT: {non_crdt_val:.2f}', 
                xy=(50, non_crdt_val), 
                xytext=(50, non_crdt_val+0.3),
                fontsize=12, 
                color='red',
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'throughput_comparison.png'), dpi=300)
    print(f"Saved throughput comparison plot to {FIGURES_DIR}")

def plot_improvement_percentage(df):
    """Plot percentage improvement from using CRDT across node counts"""
    plt.figure(figsize=(14, 8))
    
    # Calculate improvements for each metric across node counts
    improvements = []
    
    for node_count in sorted(df['node_count'].unique()):
        node_df = df[df['node_count'] == node_count]
        avg_node_df = calculate_averages(node_df)
        
        # Get the average values for CRDT and non-CRDT
        crdt_vals = avg_node_df[avg_node_df['use_crdt'] == True]
        non_crdt_vals = avg_node_df[avg_node_df['use_crdt'] == False]
        
        if len(crdt_vals) > 0 and len(non_crdt_vals) > 0:
            # For latency and hop count, improvement is percentage reduction (lower is better)
            latency_improve = ((non_crdt_vals['average_latency'].values[0] - crdt_vals['average_latency'].values[0]) / 
                              non_crdt_vals['average_latency'].values[0]) * 100
            
            hop_count_improve = ((non_crdt_vals['average_hop_count'].values[0] - crdt_vals['average_hop_count'].values[0]) / 
                                non_crdt_vals['average_hop_count'].values[0]) * 100
            
            # For throughput and success rate, improvement is percentage increase (higher is better)
            throughput_improve = ((crdt_vals['average_throughput'].values[0] - non_crdt_vals['average_throughput'].values[0]) / 
                                 non_crdt_vals['average_throughput'].values[0]) * 100
            
            success_rate_improve = ((crdt_vals['success_rate'].values[0] - non_crdt_vals['success_rate'].values[0]) / 
                                   non_crdt_vals['success_rate'].values[0]) * 100
            
            improvements.append({
                'node_count': node_count,
                'Latency': latency_improve,
                'Hop Count': hop_count_improve,
                'Throughput': throughput_improve,
                'Success Rate': success_rate_improve
            })
    
    # Convert to DataFrame
    imp_df = pd.DataFrame(improvements)
    
    # Melt for easier plotting
    melted_df = pd.melt(imp_df, id_vars=['node_count'], var_name='Metric', value_name='Improvement %')
    
    # Create a more differentiated color palette for the metrics
    color_map = {
        'Latency': '#4CAF50',      # Green
        'Hop Count': '#2196F3',    # Blue
        'Throughput': '#FF9800',   # Orange
        'Success Rate': '#9C27B0'  # Purple
    }
    
    # Create the plot
    ax = plt.subplot(111)
    bars = sns.barplot(
        data=melted_df,
        x='node_count',
        y='Improvement %',
        hue='Metric',
        palette=color_map,
        ax=ax
    )
    
    # Add value labels on top of bars
    for container in bars.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10)
    
    plt.title('CRDT Performance Improvement by Metric and Node Count', fontsize=16)
    plt.xlabel('Number of Nodes', fontsize=14)
    plt.ylabel('Improvement Percentage (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Metric', fontsize=12, title_fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at 0%
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Ensure x-axis labels are correct node counts
    plt.xticks(range(len(imp_df['node_count'])), imp_df['node_count'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'improvement_percentage.png'), dpi=300)
    print(f"Saved improvement percentage plot to {FIGURES_DIR}")

def plot_all_metrics_radar(df):
    """Create a radar chart comparing all metrics for CRDT vs non-CRDT"""
    # For radar chart, we'll use the 30-node configuration as representative
    radar_df = df[df['node_count'] == 30].copy()
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
    
    # Set up the radar chart
    categories = ['Latency\n(lower is better)', 'Throughput\n(higher is better)', 
                 'Success Rate\n(higher is better)', 'Hop Count\n(lower is better)']
    
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add category labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], fontsize=10)
    plt.ylim(0, 1)
    
    # Plot data for each CRDT setting
    for use_crdt in [True, False]:
        # Get the normalized metrics for this setting
        values = [avg_radar_df[avg_radar_df['use_crdt'] == use_crdt][f'{metric}_norm'].values[0] for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Plot the data
        color = 'green' if use_crdt else 'red'
        label = 'CRDT' if use_crdt else 'Non-CRDT'
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=color, label=label)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    plt.title('Performance Comparison at 30 Nodes (Higher is Better)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'radar_comparison.png'), dpi=300)
    print(f"Saved radar comparison plot to {FIGURES_DIR}")

def create_summary_table(df):
    """Create a summary table of all metrics"""
    # Group by node count and CRDT usage, calculate averages
    summary_df = calculate_averages(df)
    
    # Pivot the table to have metrics as rows and node counts as columns
    # Split by CRDT usage
    metrics = ['average_latency', 'average_hop_count', 'success_rate', 'average_throughput']
    
    # Create a multi-index DataFrame for better organization
    result_list = []
    
    for metric in metrics:
        for use_crdt in [True, False]:
            metric_data = summary_df[summary_df['use_crdt'] == use_crdt][['node_count', metric]]
            metric_data = metric_data.set_index('node_count').T
            
            # Format the metric name
            metric_name = {
                'average_latency': 'Latency (s)',
                'average_hop_count': 'Hop Count',
                'success_rate': 'Success Rate',
                'average_throughput': 'Throughput (pkts/s)'
            }.get(metric, metric)
            
            # Add CRDT usage information
            approach = 'CRDT' if use_crdt else 'Non-CRDT'
            
            # Create a Series with a multi-index
            for _, row in metric_data.iterrows():
                result_list.append({
                    'Metric': metric_name,
                    'Approach': approach,
                    **{str(col): row[col] for col in row.index}
                })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result_list)
    
    # Calculate the improvement percentage for each metric and node count
    improvement_list = []
    metrics_display = {
        'Latency (s)': 'Latency Reduction',
        'Hop Count': 'Hop Count Reduction',
        'Success Rate': 'Success Rate Improvement',
        'Throughput (pkts/s)': 'Throughput Improvement'
    }
    
    for metric in metrics_display.keys():
        crdt_rows = result_df[(result_df['Metric'] == metric) & (result_df['Approach'] == 'CRDT')]
        non_crdt_rows = result_df[(result_df['Metric'] == metric) & (result_df['Approach'] == 'Non-CRDT')]
        
        if not crdt_rows.empty and not non_crdt_rows.empty:
            improvement_row = {'Metric': metrics_display[metric], 'Approach': 'Improvement (%)'}
            
            for node_count in [str(n) for n in sorted(df['node_count'].unique())]:
                crdt_val = crdt_rows[node_count].values[0]
                non_crdt_val = non_crdt_rows[node_count].values[0]
                
                # For latency and hop count, improvement is reduction (lower is better)
                if metric in ['Latency (s)', 'Hop Count']:
                    improvement = ((non_crdt_val - crdt_val) / non_crdt_val) * 100
                # For throughput and success rate, improvement is increase (higher is better)
                else:
                    improvement = ((crdt_val - non_crdt_val) / non_crdt_val) * 100
                
                improvement_row[node_count] = improvement
            
            improvement_list.append(improvement_row)
    
    # Add improvement rows to the result DataFrame
    result_df = pd.concat([result_df, pd.DataFrame(improvement_list)], ignore_index=True)
    
    # Format values in the DataFrame
    for col in result_df.columns:
        if col not in ['Metric', 'Approach']:
            result_df[col] = result_df.apply(
                lambda row: f"{row[col]:.2f}%" if row['Approach'] == 'Improvement (%)' else (
                    f"{row[col]:.2f}" if row['Metric'] != 'Success Rate' else f"{row[col]:.2%}"
                ),
                axis=1
            )
    
    # Save as CSV
    result_df.to_csv(os.path.join(DATA_DIR, 'performance_summary.csv'), index=False)
    print(f"Saved performance summary to {DATA_DIR}/performance_summary.csv")
    
    # Create HTML table with styling for better visualization
    styled_df = result_df.style
    
    # Apply conditional formatting to highlight improvements
    def highlight_improvements(val):
        if isinstance(val, str) and val.endswith('%'):
            val_num = float(val.rstrip('%'))
            color = 'green' if val_num > 0 else ('red' if val_num < 0 else 'black')
            return f'color: {color}; font-weight: bold;'
        return ''
    
    for col in result_df.columns:
        if col not in ['Metric', 'Approach']:
            styled_df = styled_df.applymap(highlight_improvements, subset=pd.IndexSlice[:, col])
    
    # Apply general styling
    styled_df = styled_df.set_properties(**{
        'border': '1px solid #ddd',
        'padding': '8px',
        'text-align': 'right'
    })
    
    # Special styling for headers
    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f2f2f2'), 
                                     ('color', 'black'),
                                     ('font-weight', 'bold'),
                                     ('border', '1px solid #ddd'),
                                     ('padding', '8px'),
                                     ('text-align', 'center')]},
        {'selector': 'caption', 'props': [('caption-side', 'top'), 
                                          ('font-size', '1.5em'),
                                          ('font-weight', 'bold')]}
    ])
    
    # Add a caption
    styled_df = styled_df.set_caption('CRDT vs. Non-CRDT Performance Summary')
    
    # Save as HTML
    styled_df.to_html(os.path.join(DATA_DIR, 'performance_summary.html'))
    print(f"Saved formatted performance summary to {DATA_DIR}/performance_summary.html")
    
    return result_df

def main():
    # Load the generated performance data
    print("Loading performance data...")
    df = load_data()
    
    # Create output directories
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Generate all plots
    print("Generating plots...")
    plot_latency_comparison(df)         # Now shows delay (5-25s vs 6-45s)
    plot_hop_count_comparison(df)       # Now shows hop counts (1.5-9 vs 1.6-14)
    plot_success_rate_comparison(df)    # Now shows success rates (98%-89% vs 95%-78%)
    plot_throughput_comparison(df)      # Now shows throughput (4-1.4 vs 3.9-0.9 pkts/s)
    plot_all_metrics_radar(df)          # Radar chart comparing all metrics
    plot_improvement_percentage(df)     # Bar chart showing percentage improvement
    
    # Create summary table
    print("Creating summary tables...")
    create_summary_table(df)
    
    print("Visualization complete! All plots saved to:", FIGURES_DIR)

if __name__ == "__main__":
    main() 