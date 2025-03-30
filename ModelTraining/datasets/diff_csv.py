import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(file_path)

def find_device_columns(df: pd.DataFrame) -> List[str]:
    """Find all device columns that end with '_on'."""
    return [col for col in df.columns if col.endswith('_on')]

def find_longest_matching_sequences(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Find the longest matching sequence for each device between two datasets.
    
    Returns:
        Dictionary with device names as keys and dictionaries containing:
        - longest_sequence: length of the longest matching sequence
        - start_idx1: starting index in df1
        - start_idx2: starting index in df2
        - percentage: percentage of the dataset that matches
    """
    results = {}
    
    # Get all device columns from both dataframes
    device_cols = list(set(find_device_columns(df1)) | set(find_device_columns(df2)))
    
    for device_col in device_cols:
        device = device_col.replace('_on', '')
        
        # Check if this device exists in both dataframes
        if device_col not in df1.columns or device_col not in df2.columns:
            results[device] = {
                "longest_sequence": 0,
                "start_idx1": None,
                "start_idx2": None,
                "percentage": 0
            }
            continue
            
        # Extract device states
        states1 = df1[device_col].to_numpy()
        states2 = df2[device_col].to_numpy()
        
        # Find longest matching sequence
        longest = 0
        start1 = -1
        start2 = -1
        
        for i in range(len(states1)):
            for j in range(len(states2)):
                length = 0
                while (i + length < len(states1) and 
                       j + length < len(states2) and 
                       states1[i + length] == states2[j + length]):
                    length += 1
                
                if length > longest:
                    longest = length
                    start1 = i
                    start2 = j
        
        # Calculate percentage of the smaller dataset
        smaller_len = min(len(states1), len(states2))
        percentage = (longest / smaller_len * 100) if smaller_len > 0 else 0
        
        results[device] = {
            "longest_sequence": longest,
            "start_idx1": start1,
            "start_idx2": start2,
            "percentage": percentage
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Find the longest matching sequence between two CSV files.')
    parser.add_argument('file1', type=str, help='Path to the first CSV file')
    parser.add_argument('file2', type=str, help='Path to the second CSV file')
    args = parser.parse_args()
    
    print(f"Comparing {args.file1} and {args.file2}...")
    
    df1 = load_csv(args.file1)
    df2 = load_csv(args.file2)
    
    results = find_longest_matching_sequences(df1, df2)
    
    print("\nResults:")
    print("=" * 80)
    print(f"{'Device':<20} {'Longest Sequence':<20} {'Match %':<10} {'Start in File 1':<15} {'Start in File 2':<15}")
    print("-" * 80)
    
    # Sort results by matching percentage (descending)
    for device, data in sorted(results.items(), key=lambda x: x[1]['percentage'], reverse=True):
        print(f"{device:<20} {data['longest_sequence']:<20} {data['percentage']:.2f}% {data['start_idx1']:<15} {data['start_idx2']:<15}")
    
    # Calculate overall statistics
    total_devices = len(results)
    avg_percentage = sum(data['percentage'] for data in results.values()) / total_devices if total_devices > 0 else 0
    max_match = max((data['percentage'] for data in results.values()), default=0)
    
    print("=" * 80)
    print(f"Total devices: {total_devices}")
    print(f"Average matching percentage: {avg_percentage:.2f}%")
    print(f"Maximum matching percentage: {max_match:.2f}%")

if __name__ == "__main__":
    main()
