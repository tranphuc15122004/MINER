#!/usr/bin/env python3
"""Convert a production-style prediction file to rank format.

Input format (per line):
<impression_id> [v1, v2, v3, ...]

Output format (per line):
<impression_id> [r1,r2,r3,...]

Ranks are assigned so that the highest value gets rank 1.
"""
import argparse
import ast
import sys


def convert_line(line: str):
    line = line.strip()
    if not line:
        return None
    # Split at first space
    try:
        imp_id_str, list_str = line.split(' ', 1)
    except ValueError:
        raise ValueError(f'Unexpected line format: {line!r}')

    imp_id = imp_id_str.strip()
    # Safely parse the list using ast.literal_eval
    try:
        values = ast.literal_eval(list_str.strip())
    except Exception as e:
        raise ValueError(f'Could not parse list on line: {line!r}\n{e}')

    # Ensure it's a sequence
    if not hasattr(values, '__len__'):
        raise ValueError(f'Parsed values is not a list/sequence: {values!r}')

    # Compute ranks: highest value -> rank 1
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0] * len(values)
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank

    return imp_id, ranks


def main():
    parser = argparse.ArgumentParser(description='Convert prod prediction file to rank file')
    parser.add_argument('input', help='Input prod file (e.g., checkpoint/prediction_prod_Son.txt)')
    parser.add_argument('output', help='Output rank file (e.g., checkpoint/prediction_rank_Son.txt)')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            parsed = convert_line(line)
            if parsed is None:
                continue
            imp_id, ranks = parsed
            out_str = ','.join(map(str, ranks))
            fout.write(f'{imp_id} [{out_str}]\n')


if __name__ == '__main__':
    main()
