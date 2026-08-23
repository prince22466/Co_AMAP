Dataset Description
The full dataset is hosted on HuggingFace: UkrainianCatholicUniversity/rukopys

from datasets import load_dataset
ds = load_dataset("UkrainianCatholicUniversity/rukopys")
Files
train/images/ — JPEG images with human-verified annotations
train/metadata.jsonl — one JSON record per image: bounding boxes, region types, and transcribed text
silver/images/ — JPEG images with auto-generated annotations (for self-training)
silver/metadata.jsonl — same format as train
test/images/ — JPEG images; no annotations provided
test/metadata.jsonl — image filename, dimensions, and source only
sample_submission.csv — a valid submission file with empty region lists for all test images
Columns in metadata.jsonl
Field	Description
file_name	Relative path to the image file (e.g. images/abc123.jpg)
image_width	Image width in pixels
image_height	Image height in pixels
source	Data source: dictation, archive, university, school
annotation_source	Annotation tier: annotator (Keymakr), volunteer, auto (silver only)
regions	List of annotated regions (train/silver only; null for test)
Each region in regions:

Field	Description
bbox	[x1, y1, x2, y2] — pixel coordinates, top-left origin
type	Region type — see the table below
language	uk (Ukrainian) or other
legibility	legible or illegible
text	Transcribed text; empty for image, graph, illegible, and language=other regions
Region types
Type	What it covers	Transcription
handwritten	Handwritten text line	Exact text, 1 bbox per line
printed	Printed/typed text line (textbook headers, captions, typeset stamps)	Exact text, 1 bbox per line
formula	Standalone math or chemistry expression	LaTeX (\frac{1}{x}, \sqrt{2}, \sin\alpha, etc.)
table	Full table	Pipe-separated values, one row per line
annotation	Teacher marks, grades, page numbering, marginalia	Short text
image	Photo-like content: drawings, illustrations, stamps, seals	Empty string ""
graph	Geometric drawings: figures, coordinate systems, plots, charts	Empty string ""
image and graph regions, together with language=other and legibility=illegible GT regions, are excluded from both Region CER and Page CER scoring — predict an empty string for image and graph. They still contribute to Detection F1 (15%) and Classification Accuracy (5%), so detecting and classifying them correctly is rewarded.

Special transcription markers
Used in train/silver GT, can also appear in your predictions:

Marker	Meaning
~~word~~	Strikethrough text
~~old~~{new}	Strikethrough with correction
[illegible]	Unreadable word within an otherwise legible line
Both ground-truth and predicted text go through the same normalization pipeline before CER — see the official metric notebook for the full ruleset (LaTeX commands, Latin↔Cyrillic lookalikes, Unicode super/subscripts, dash/quote normalization, etc.).