Evaluation
Submissions are scored using a composite metric that evaluates three aspects of handwritten document understanding: region detection, type classification, and text transcription.

Score = 0.15 × Detection_F1 + 0.05 × ClassAcc + 0.30 × (1 − CER) + 0.50 × (1 − PageCER)
Score Components
Detection F1 (15%) — Bounding box detection quality, type-agnostic. A predicted box is counted as a true positive if its IoU with a ground-truth box is ≥ 0.5. Precision, recall and F1 are computed globally across all images.

Classification Accuracy (5%) — Among all IoU-matched region pairs, the fraction where the predicted type matches the ground-truth type. Region types: handwritten, printed, formula, table, annotation, image, graph.

Character Error Rate — per-region (30%) — Levenshtein distance divided by ground-truth text length, averaged across all matched scorable regions. A region is scorable when the ground-truth has language=uk, legibility=legible, and type is not image or graph.

Page CER (50%) — Full-page text comparison. Ground-truth and predicted regions are independently sorted by reading order (top-to-bottom, left-to-right) and concatenated into a single string. Page CER is the Levenshtein distance between the two strings divided by the ground-truth string length, averaged across all test images. This component is agnostic to bbox granularity — it rewards correct page text even when individual box boundaries are imprecise.

Text Normalization
Before CER comparison, within the metric function, both ground-truth and predicted text are normalized identically:

Step	Rule
Strikethrough	~~old~~{new} → new; ~~text~~ → text
LaTeX symbols	\alpha → α, \cdot → ·, \rightarrow → →, etc.
Latin/Cyrillic lookalikes	Latin c, o, p, x → Cyrillic equivalents
Dashes	em-dash, en-dash → hyphen -
Quotes	«»"" → ", '' → '
Superscripts/subscripts	x² → x^2, H₂ → H_2
LaTeX braces	x_{3} → x_3
Whitespace	collapse, strip
Submission File
For each image in the test set, predict a list of regions. The submission is a CSV file with two columns:

image,regions
abc123.jpg,"[{""bbox"":[50,100,850,130],""type"":""handwritten"",""text"":""Доброго ранку""}]"
def456.jpg,[]
image — test image filename (e.g. abc123.jpg)
regions — JSON-encoded list of detected regions; use [] for images with no detections
Each region object requires three fields:

Field	Type	Description
bbox	[x1, y1, x2, y2]	Pixel coordinates, top-left origin
type	string	One of: handwritten, printed, formula, table, annotation, image, graph
text	string	Transcribed text; empty string for image and graph regions
All test images must be present in the submission. Missing images will raise a scoring error.

Full scoring code, text normalization rules, and a local debugger:
official-evaluation-metric-text-normalization.ipynb