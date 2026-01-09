#!/usr/bin/env python3
"""
Script to verify that prediction_prod.txt has correct number of elements
matching the number of candidates in behaviors.tsv

Format:
- behaviors.tsv: Tab-separated file with columns:
  [impression_id, user_id, timestamp, history, candidates]
  where candidates is space-separated list of news IDs
  
- prediction_prod.txt: Lines with format:
  impression_id [score1,score2,...,scoreN]
  where N should equal the number of candidates for that impression
"""

import sys
import json
from pathlib import Path


def parse_behaviors_file(behaviors_path):
    """
    Parse behaviors.tsv and return dict of impression_id -> num_candidates
    
    Format of behaviors.tsv:
    impression_id \t user_id \t timestamp \t history \t candidates
    """
    impression_candidates = {}
    
    with open(behaviors_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 5:
                print(f"Warning: Line {line_num} has {len(parts)} fields (expected 5)")
                continue
            
            impression_id = parts[0]
            candidates_str = parts[4]
            
            # Count number of candidates (space-separated news IDs)
            if candidates_str:
                candidates = candidates_str.split()
                num_candidates = len(candidates)
            else:
                num_candidates = 0
            
            impression_candidates[impression_id] = num_candidates
    
    return impression_candidates


def parse_prediction_file(prediction_path):
    """
    Parse prediction_prod.txt and return prediction info
    
    Format: impression_id [score1,score2,...,scoreN]
    
    Returns:
        tuple: (impression_predictions, invalid_scores)
        - impression_predictions: dict of impression_id -> num_predictions
        - invalid_scores: list of (impression_id, line_num, invalid_values)
    """
    impression_predictions = {}
    invalid_scores = []
    
    with open(prediction_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(' ', 1)
            if len(parts) != 2:
                print(f"Warning: Line {line_num} in prediction file has unexpected format")
                continue
            
            impression_id = parts[0]
            scores_str = parts[1]
            
            # Parse the array of scores
            try:
                scores = json.loads(scores_str)
                num_predictions = len(scores)
                
                # Check if all scores are in range [0, 1]
                out_of_range = []
                for i, score in enumerate(scores):
                    if not isinstance(score, (int, float)):
                        out_of_range.append((i, score, "not a number"))
                    elif score < 0 or score > 1:
                        out_of_range.append((i, score, "out of range [0,1]"))
                
                if out_of_range:
                    invalid_scores.append((impression_id, line_num, out_of_range))
                    
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} - Failed to parse scores: {e}")
                continue
            
            impression_predictions[impression_id] = num_predictions
    
    return impression_predictions, invalid_scores


def verify_predictions(behaviors_path, prediction_path):
    """
    Verify that predictions match the expected number of candidates
    """
    print(f"Reading behaviors file: {behaviors_path}")
    impression_candidates = parse_behaviors_file(behaviors_path)
    print(f"Found {len(impression_candidates)} impressions in behaviors file")
    
    print(f"\nReading prediction file: {prediction_path}")
    impression_predictions, invalid_scores = parse_prediction_file(prediction_path)
    print(f"Found {len(impression_predictions)} impressions in prediction file")
    
    # Check for missing impressions
    missing_in_predictions = set(impression_candidates.keys()) - set(impression_predictions.keys())
    extra_in_predictions = set(impression_predictions.keys()) - set(impression_candidates.keys())
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    if missing_in_predictions:
        print(f"\n⚠️  WARNING: {len(missing_in_predictions)} impressions in behaviors but NOT in predictions:")
        for imp_id in sorted(missing_in_predictions, key=int)[:10]:
            print(f"  - Impression {imp_id} (expected {impression_candidates[imp_id]} predictions)")
        if len(missing_in_predictions) > 10:
            print(f"  ... and {len(missing_in_predictions) - 10} more")
    
    if extra_in_predictions:
        print(f"\n⚠️  WARNING: {len(extra_in_predictions)} impressions in predictions but NOT in behaviors:")
        for imp_id in sorted(extra_in_predictions, key=int)[:10]:
            print(f"  - Impression {imp_id}")
        if len(extra_in_predictions) > 10:
            print(f"  ... and {len(extra_in_predictions) - 10} more")
    
    # Check for mismatches in number of predictions
    mismatches = []
    for imp_id in impression_predictions:
        if imp_id in impression_candidates:
            expected = impression_candidates[imp_id]
            actual = impression_predictions[imp_id]
            if expected != actual:
                mismatches.append((imp_id, expected, actual))
    
    if mismatches:
        print(f"\n❌ ERROR: {len(mismatches)} impressions have MISMATCHED prediction counts:")
        for imp_id, expected, actual in sorted(mismatches, key=lambda x: int(x[0]))[:20]:
            print(f"  - Impression {imp_id}: expected {expected} predictions, got {actual}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
    else:
        print("\n✅ SUCCESS: All impressions have correct number of predictions!")
    
    # Check for invalid score values
    if invalid_scores:
        print(f"\n❌ ERROR: {len(invalid_scores)} impressions have INVALID score values (not in [0,1]):")
        for imp_id, line_num, out_of_range in invalid_scores[:10]:
            print(f"  - Impression {imp_id} (line {line_num}):")
            for idx, value, reason in out_of_range[:5]:
                print(f"      Score[{idx}] = {value} ({reason})")
            if len(out_of_range) > 5:
                print(f"      ... and {len(out_of_range) - 5} more invalid scores")
        if len(invalid_scores) > 10:
            print(f"  ... and {len(invalid_scores) - 10} more impressions with invalid scores")
    else:
        print("\n✅ SUCCESS: All scores are valid (in range [0, 1])!")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total impressions in behaviors: {len(impression_candidates)}")
    print(f"Total impressions in predictions: {len(impression_predictions)}")
    print(f"Missing impressions: {len(missing_in_predictions)}")
    print(f"Extra impressions: {len(extra_in_predictions)}")
    print(f"Mismatched counts: {len(mismatches)}")
    print(f"Invalid score values: {len(invalid_scores)}")
    
    if not missing_in_predictions and not extra_in_predictions and not mismatches and not invalid_scores:
        print("\n🎉 PERFECT MATCH! Prediction file is valid.")
        return True
    else:
        print("\n⚠️  VALIDATION FAILED! Please check the errors above.")
        return False


def main():
    # Default paths
    behaviors_path = Path("/home/tuantb/MINER/data/MINDlarge_test/MINDlarge_test/behaviors.tsv")
    prediction_path = Path("prediction_score_22.txt")
    
    # Allow custom paths from command line
    if len(sys.argv) > 1:
        prediction_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        behaviors_path = Path(sys.argv[2])
    
    # Check files exist
    if not behaviors_path.exists():
        print(f"Error: Behaviors file not found: {behaviors_path}")
        sys.exit(1)
    if not prediction_path.exists():
        print(f"Error: Prediction file not found: {prediction_path}")
        sys.exit(1)
    
    # Run verification
    success = verify_predictions(str(behaviors_path), str(prediction_path))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
