import random
from pathlib import Path


input_file = Path("raw/2024Q1.csv")
output_file = Path("raw_process/2024Q1_30.csv")
sample_rate = 0.3


line_count = sum(1 for _ in input_file.open("r", encoding="utf-8"))
sample_size = int(line_count * sample_rate)
selected_lines = set(random.sample(range(line_count), sample_size))

output_file.parent.mkdir(parents=True, exist_ok=True)

with input_file.open("r", encoding="utf-8") as src, output_file.open(
    "w", encoding="utf-8"
) as dst:
    for line_number, line in enumerate(src):
        if line_number in selected_lines:
            dst.write(line)

print(f"Total lines: {line_count}")
print(f"Copied lines: {sample_size}")
print(f"Output file: {output_file}")
