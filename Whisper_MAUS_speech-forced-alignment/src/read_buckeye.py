#Reading the Buckeye gold data

import csv

def extract_buckeye(filepath):
    segments = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            start, end, label = parts[0], parts[1], parts[2]
            label = label.strip().lower()

            if label in ("", "sil", "<sil>", "<noise>", "b_trans", "<laugh>"):
                continue

            segments.append({
                "word": label,
                "start": float(start),
                "end": float(end)
            })

    return segments

def save_to_csv(segments, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "start", "end"])
        writer.writeheader()
        writer.writerows(segments)

segments = extract_buckeye("buckeye.txt")
save_to_csv(segments, "results/buckeye_segments.csv")
print(f"Extracted {len(segments)} segments")