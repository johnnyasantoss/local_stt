#!/usr/bin/env python3
import re
import sys

# Regular expression to match the input format: [HH:MM.SSS -> HH:MM.SSS] Text
pattern = re.compile(r'\[(\d{2}:\d{2}\.\d{3}) -> (\d{2}:\d{2}\.\d{3})\] (.*)')

# Write the WebVTT header
print('WEBVTT')
print('')  # Blank line after header

cue_num = 1

# Read from stdin line by line
for line in sys.stdin:
    line = line.strip()
    if not line:  # Skip empty lines
        continue
    
    match = pattern.match(line)
    if match:
        start, end, text = match.groups()
        # Output the cue number
        print(cue_num)
        # Output the timestamp in VTT format
        print(f'{start} --> {end}')
        # Output the text (preserving any trailing spaces if present)
        print(text)
        print('')  # Blank line to separate cues
        cue_num += 1

# Note: This script reads from standard input (e.g., cat input.txt | python script.py > output.vtt)
# It assumes each cue is on a single line in the input file.
# If your input has multi-line text per cue, you may need to adjust the parsing logic.
