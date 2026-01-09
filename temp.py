#!/usr/bin/env python3
import sys, os

INPUT_PATH = "data/MINDlarge_test/MINDlarge_test/behaviors.tsv"
OUTPUT_PATH = "data/MINDlarge_test/MINDlarge_test/small_behaviors.tsv"
START_IMPRESSION_ID = 2031903  # Bắt đầu từ impression ID này

if INPUT_PATH and OUTPUT_PATH:
    inp, outp = INPUT_PATH, OUTPUT_PATH
else:
    if len(sys.argv) != 3:
        print("Usage: temp.py input.tsv output.tsv", file=sys.stderr)
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]

os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)

with open(inp, 'r', encoding='utf-8') as fin, \
     open(outp, 'w', encoding='utf-8') as fout:
    # Ghi header
    header = fin.readline()
    fout.write(header)

    start_found = False
    written_count = 0
    
    for line in fin:
        impression_id = int(line.split('\t')[0])
        
        # Bắt đầu ghi từ impression ID 717733 trở đi
        if impression_id >= START_IMPRESSION_ID:
            start_found = True
        
        if start_found:
            fout.write(line)
            written_count += 1

print(f"Đã ghi {written_count} dòng bắt đầu từ impression ID {START_IMPRESSION_ID}")
