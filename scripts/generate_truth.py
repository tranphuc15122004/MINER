#!/usr/bin/env python3
"""Generate truth.txt for phase2 evaluator from MIND behaviors.tsv

Writes lines of the form:
<impression_id> <json_array_of_labels>

Usage: python phase2/generate_truth.py --behaviors data/valid/behaviors.tsv --out-dir phase2/ref
"""
import os
import argparse
import json


def parse_behaviors_line(line: str):
    # behaviors.tsv columns: impression_id \t user_id \t time \t history \t impressions
    parts = line.rstrip('\n').split('\t')
    if len(parts) < 5:
        return None
    impid = parts[0]
    impressions = parts[4].strip()
    if impressions == '':
        # no impressions (masked), represent as empty list
        return impid, []

    # impressions are like: N28682-0 N48740-0 ...
    items = impressions.split()
    labels = []
    for it in items:
        # some items might be like N12345-0 or just N12345 (rare)
        if '-' in it:
            try:
                label = int(it.rsplit('-', 1)[1])
            except Exception:
                label = 0
        else:
            label = 0
        labels.append(label)

    return impid, labels


def generate_truth(behaviors_path: str, out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'truth.txt')
    with open(behaviors_path, 'r', encoding='utf-8') as bf, open(out_path, 'w', encoding='utf-8') as of:
        for line in bf:
            parsed = parse_behaviors_line(line)
            if parsed is None:
                continue
            impid, labels = parsed
            # write as: <impression_id> <json_array>
            of.write(f"{impid} {json.dumps(labels, separators=(',', ':'))}\n")

    print(f"Wrote truth file: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--behaviors', default='data/valid/behaviors.tsv')
    parser.add_argument('--out-dir', default='phase2/ref')
    args = parser.parse_args()

    if not os.path.isfile(args.behaviors):
        raise FileNotFoundError(f"Behaviors file not found: {args.behaviors}")

    generate_truth(args.behaviors, args.out_dir)


if __name__ == '__main__':
    main()
