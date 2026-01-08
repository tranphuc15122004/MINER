#!/usr/bin/env python3
import sys, math, os

INPUT_PATH = "data/MINDlarge_test/MINDlarge_test/behaviors.tsv"
OUTPUT_PATH = "data/MINDlarge_test/MINDlarge_test/small_behaviors.tsv"
FRACTION = 0.001  # 0.1%

if INPUT_PATH and OUTPUT_PATH:
    inp, outp = INPUT_PATH, OUTPUT_PATH
else:
    if len(sys.argv) != 3:
        print("Usage: temp.py input.tsv output.tsv", file=sys.stderr)
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]

# Đếm số dòng dữ liệu (không tính header)
with open(inp, 'r', encoding='utf-8') as f:
    header = f.readline()
    total = sum(1 for _ in f)

k = max(1, math.ceil(total * FRACTION))

os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)

with open(inp, 'r', encoding='utf-8') as fin, \
     open(outp, 'w', encoding='utf-8') as fout:
    fout.write(header)
    next(fin)  # skip header

    for i, line in enumerate(fin):
        if i >= k:
            break
        fout.write(line)
