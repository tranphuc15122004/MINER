#!/usr/bin/env python3
"""
Script to merge two prediction files into one.

Format of prediction file:
impression_id [score1,score2,...,scoreN]

The script will:
1. Read both prediction files
2. Combine them (second file's impressions are appended after first file)
3. Handle duplicates (can choose to keep first, last, or error on duplicates)
4. Write to output file
"""

import sys
import json
from pathlib import Path
from collections import OrderedDict


def read_prediction_file(file_path):
    """
    Read a prediction file and return OrderedDict of impression_id -> scores
    
    Args:
        file_path: Path to prediction file
    
    Returns:
        OrderedDict: impression_id -> [scores]
    """
    predictions = OrderedDict()
    
    print(f"Reading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(' ', 1)
            if len(parts) != 2:
                print(f"Warning: Line {line_num} has unexpected format, skipping")
                continue
            
            impression_id = parts[0]
            scores_str = parts[1]
            
            try:
                scores = json.loads(scores_str)
                predictions[impression_id] = scores
            except json.JSONDecodeError as e:
                print(f"Error at line {line_num}: Failed to parse scores - {e}")
                continue
    
    print(f"  Loaded {len(predictions)} impressions")
    return predictions


def merge_predictions(pred1, pred2, duplicate_strategy='error'):
    """
    Merge two prediction dictionaries.
    
    Args:
        pred1: First OrderedDict of predictions
        pred2: Second OrderedDict of predictions
        duplicate_strategy: How to handle duplicates
            - 'error': Raise error if duplicate found
            - 'first': Keep prediction from first file
            - 'last': Keep prediction from second file (overwrite)
            - 'skip': Skip duplicates from second file
    
    Returns:
        OrderedDict: Merged predictions
    """
    merged = OrderedDict()
    duplicates = []
    
    # Add all from first file
    for imp_id, scores in pred1.items():
        merged[imp_id] = scores
    
    # Add from second file based on strategy
    for imp_id, scores in pred2.items():
        if imp_id in merged:
            duplicates.append(imp_id)
            if duplicate_strategy == 'error':
                raise ValueError(f"Duplicate impression_id found: {imp_id}")
            elif duplicate_strategy == 'first':
                # Keep the one from first file, skip this
                continue
            elif duplicate_strategy == 'last':
                # Overwrite with second file
                merged[imp_id] = scores
            elif duplicate_strategy == 'skip':
                # Skip duplicate
                continue
        else:
            merged[imp_id] = scores
    
    if duplicates:
        print(f"\nWarning: Found {len(duplicates)} duplicate impression_ids")
        print(f"Strategy '{duplicate_strategy}' applied")
        if len(duplicates) <= 10:
            print(f"Duplicates: {', '.join(duplicates)}")
        else:
            print(f"First 10 duplicates: {', '.join(duplicates[:10])}")
    
    return merged


def write_prediction_file(predictions, output_path):
    """
    Write predictions to output file.
    
    Args:
        predictions: OrderedDict of impression_id -> scores
        output_path: Path to output file
    """
    print(f"\nWriting to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for imp_id, scores in predictions.items():
            # Use separators without spaces to match input format: [0.1,0.2,0.3]
            scores_str = json.dumps(scores, separators=(',', ':'))
            f.write(f"{imp_id} {scores_str}\n")
    
    print(f"  Written {len(predictions)} impressions")


def main():
    if len(sys.argv) < 4:
        print("Usage: python merge_predictions.py <file1> <file2> <output> [duplicate_strategy]")
        print("\nArguments:")
        print("  file1, file2    : Input prediction files to merge")
        print("  output          : Output merged prediction file")
        print("  duplicate_strategy (optional): How to handle duplicates")
        print("                    - 'error' (default): Raise error if duplicate found")
        print("                    - 'first': Keep prediction from first file")
        print("                    - 'last': Keep prediction from second file")
        print("                    - 'skip': Skip duplicates from second file")
        print("\nExample:")
        print("  python merge_predictions.py pred1.txt pred2.txt merged.txt last")
        sys.exit(1)
    
    file1_path = Path(sys.argv[1])
    file2_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    duplicate_strategy = sys.argv[4] if len(sys.argv) > 4 else 'error'
    
    # Validate strategy
    valid_strategies = ['error', 'first', 'last', 'skip']
    if duplicate_strategy not in valid_strategies:
        print(f"Error: Invalid duplicate_strategy '{duplicate_strategy}'")
        print(f"Valid strategies: {', '.join(valid_strategies)}")
        sys.exit(1)
    
    # Check input files exist
    if not file1_path.exists():
        print(f"Error: File not found: {file1_path}")
        sys.exit(1)
    if not file2_path.exists():
        print(f"Error: File not found: {file2_path}")
        sys.exit(1)
    
    # Check output file doesn't exist (safety check)
    if output_path.exists():
        response = input(f"Warning: {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    print("="*80)
    print("MERGING PREDICTION FILES")
    print("="*80)
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print(f"Output: {output_path}")
    print(f"Duplicate strategy: {duplicate_strategy}")
    print()
    
    # Read both files
    pred1 = read_prediction_file(file1_path)
    pred2 = read_prediction_file(file2_path)
    
    # Merge
    print("\nMerging...")
    try:
        merged = merge_predictions(pred1, pred2, duplicate_strategy)
    except ValueError as e:
        print(f"\nError during merge: {e}")
        sys.exit(1)
    
    # Write output
    write_prediction_file(merged, output_path)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"File 1 impressions: {len(pred1)}")
    print(f"File 2 impressions: {len(pred2)}")
    print(f"Merged impressions: {len(merged)}")
    print(f"\n✅ Successfully merged prediction files!")


if __name__ == "__main__":
    main()
