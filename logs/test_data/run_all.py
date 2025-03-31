#!/usr/bin/env python3

import os
import subprocess
import sys
import time

def run_script(script_name, description):
    """Run a Python script and print its output"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}\n")
    
    try:
        # Run the script and capture its output
        result = subprocess.run([sys.executable, script_name], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               check=True)
        
        # Print the output
        print(result.stdout)
        
        if result.stderr:
            print("WARNINGS/ERRORS:")
            print(result.stderr)
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run all scripts for data generation and visualization"""
    start_time = time.time()
    
    print("\nCRDT vs. Non-CRDT Performance Analysis")
    print("======================================\n")
    
    # Create figures directories if they don't exist
    os.makedirs("figures", exist_ok=True)
    os.makedirs("real_data_figures", exist_ok=True)
    
    # Step 1: Generate random data
    if run_script("generate_data.py", "Generating random performance data"):
        print("✓ Random data generation complete")
    else:
        print("✗ Random data generation failed")
        return
    
    # Step 2: Visualize the random data
    if run_script("visualize_data.py", "Creating visualizations from random data"):
        print("✓ Visualization of random data complete")
    else:
        print("✗ Visualization of random data failed")
    
    # Step 3: Analyze real data and compare
    if run_script("analyze_real_data.py", "Analyzing real experiment data"):
        print("✓ Real data analysis complete")
    else:
        print("✗ Real data analysis failed")
    
    # Calculate total runtime
    runtime = time.time() - start_time
    print(f"\nAll processing completed in {runtime:.2f} seconds\n")
    
    # List generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk("."):
        if "figures" in root and files:
            for file in files:
                if file.endswith((".png", ".csv", ".html")):
                    print(f"  - {os.path.join(root, file)}")

if __name__ == "__main__":
    main() 