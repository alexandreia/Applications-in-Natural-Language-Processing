import re
import csv

# Reading a .TextGrid file (MAUS + gold)
def extract_textgrid(filepath):
    segments = []

    with open(filepath, "r") as f:
        content = f.read()

    # Find all intervals with xmin, xmax, and text
    pattern = r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"'
    matches = re.findall(pattern, content)

    for start, end, label in matches:
        label = label.strip()
        if label == "" or label == "<p:>":  # skip silences
            continue
        segments.append({
            "word": label.lower(),
            "start": float(start),
            "end": float(end)
        })

    return segments

def save_to_csv(segments, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "start", "end"])
        writer.writeheader()
        writer.writerows(segments)

segments = extract_textgrid("buckeye_maus.TextGrid")
save_to_csv(segments, "results/buckeye_segments.csv")
print(f"Extracted {len(segments)} segments")