import re
import sys
import json

def time_to_vtt(seconds):
    """Convert seconds (float) to VTT timestamp format: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.3f}"

def parse_time_str(tstr):
    """Parse the previous format timestamp (MM:SS.mmm) to total seconds."""
    parts = tstr.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid timestamp format: {tstr}")
    minutes = int(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds

# Hardcoded offsets for the 4 parts based on the split timings (in seconds) [1]
# Part 1: offset 0
# Part 2: starts at 00:47:40.250 -> 2860.250
# Part 3: starts at 01:35:20.500 -> 5720.500
# Part 4: starts at 02:23:00.750 -> 8580.750
PART_OFFSETS = [0.0, 2860.250, 5720.500, 8580.750]

# Read the entire input from stdin
full_input = sys.stdin.read().strip()

# Collect valid Groq JSON segments from lines (ignoring non-JSON lines, expecting exactly 4 valid ones)
segments_all = []
lines = full_input.splitlines()
valid_jsons = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
        if 'segments' in data and isinstance(data['segments'], list):
            valid_jsons.append(data)
    except json.JSONDecodeError:
        # Skip invalid JSON lines (e.g., appended content or headers)
        continue

if len(valid_jsons) == 4:
    # Process the 4 valid JSON responses with offsets
    for i, data in enumerate(valid_jsons):
        offset = PART_OFFSETS[i]
        for seg in data['segments']:
            if 'start' in seg and 'end' in seg and 'text' in seg:
                start_sec = float(seg['start']) + offset
                end_sec = float(seg['end']) + offset
                seg_adjusted = {
                    'start': start_sec,
                    'end': end_sec,
                    'text': seg['text'].strip()
                }
                segments_all.append(seg_adjusted)
    
    # Output combined VTT
    print('WEBVTT')
    print('')  # Blank line after header
    
    cue_num = 1
    for seg in segments_all:
        print(cue_num)
        print(f"{time_to_vtt(seg['start'])} --> {time_to_vtt(seg['end'])}")
        print(seg['text'])
        print('')  # Blank line to separate cues
        cue_num += 1
    sys.exit(0)  # Success
elif len(valid_jsons) == 1:
    # Fallback: Single Groq JSON (no offset)
    data = valid_jsons[0]
    print('WEBVTT')
    print('')  # Blank line after header
    
    cue_num = 1
    for seg in data['segments']:
        if 'start' in seg and 'end' in seg and 'text' in seg:
            start_sec = float(seg['start'])
            end_sec = float(seg['end'])
            text = seg['text'].strip()
            
            print(cue_num)
            print(f"{time_to_vtt(start_sec)} --> {time_to_vtt(end_sec)}")
            print(text)
            print('')  # Blank line to separate cues
            cue_num += 1
    sys.exit(0)
else:
    # If not exactly 4 or 1 valid JSONs, try single full JSON parse
    try:
        data = json.loads(full_input)
        if 'segments' in data and isinstance(data['segments'], list):
            print('WEBVTT')
            print('')  # Blank line after header
            
            cue_num = 1
            for seg in data['segments']:
                if 'start' in seg and 'end' in seg and 'text' in seg:
                    start_sec = float(seg['start'])
                    end_sec = float(seg['end'])
                    text = seg['text'].strip()
                    
                    print(cue_num)
                    print(f"{time_to_vtt(start_sec)} --> {time_to_vtt(end_sec)}")
                    print(text)
                    print('')  # Blank line to separate cues
                    cue_num += 1
            sys.exit(0)
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

# Final fallback: Previous regex-based parsing
lines = full_input.splitlines()
pattern = re.compile(r'\[(\d{2}:\d{2}\.\d{3}) -> (\d{2}:\d{2}\.\d{3})\] (.*)')

print('WEBVTT')
print('')  # Blank line after header

cue_num = 1
for line in lines:
    line = line.strip()
    if not line:  # Skip empty lines
        continue
    
    match = pattern.match(line)
    if match:
        start_str, end_str, text = match.groups()
        try:
            start_sec = parse_time_str(start_str)
            end_sec = parse_time_str(end_str)
        except ValueError:
            continue  # Skip invalid timestamps
        
        print(cue_num)
        print(f"{time_to_vtt(start_sec)} --> {time_to_vtt(end_sec)}")
        print(text.strip())
        print('')  # Blank line to separate cues
        cue_num += 1

# Usage: cat output.jsonl | python3 groq-to-vtt.py > output.vtt
# This now handles extra lines in the JSONL by collecting only valid 'segments'-containing JSONs.
# If you have a different number of parts or offsets, update PART_OFFSETS [1].
# Test with: head -n 4 output.jsonl | python3 groq-to-vtt.py (if the responses are the first 4 lines).
