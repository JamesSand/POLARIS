#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert all parquet files in benchmarks/ to jsonl format.
Each line in the output jsonl contains: example_id, prompt, and ground_truth.
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def convert_parquet_to_jsonl(parquet_path, output_path):
    """Convert a single parquet file to jsonl format."""
    df = pd.read_parquet(parquet_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(len(df)):
            # Extract prompt and ground truth
            prompt = df.at[i, "prompt"]
            gt = df.at[i, "reward_model"]["ground_truth"]
            
            # Convert to serializable format
            prompt = convert_to_serializable(prompt)
            gt = convert_to_serializable(gt)
            
            # Create output record
            record = {
                "example_id": i,
                "prompt": prompt,
                "ground_truth": gt
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Converted: {parquet_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert parquet files to jsonl format")
    parser.add_argument("--input_dir", type=str, 
                       default="benchmarks/processed",
                       help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str,
                       default="benchmarks/jsonl_processed",
                       help="Directory to save jsonl files")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = list(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Convert each file
    for parquet_path in parquet_files:
        # Generate output filename
        jsonl_filename = parquet_path.stem + ".jsonl"
        output_path = output_dir / jsonl_filename
        
        convert_parquet_to_jsonl(parquet_path, output_path)
    
    print(f"\nAll files converted successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()




