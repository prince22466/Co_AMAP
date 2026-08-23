# Handwritten to Data: 
https://www.kaggle.com/competitions/handwritten-to-data/overview

Overview
Goal of the challenge
As Ukraine continues its digital transformation, working with handwritten documents remains a major bottleneck. There is still a lack of open and straightforward tools for processing Ukrainian handwritten materials. Most existing solutions are either closed or built from fragmented components that do not scale well. This slows down the launch and development of public digital services, such as ePermit.

In this challenge, participants will build AI solutions to recognize Ukrainian handwritten documents — applications, certificates, logs, signatures, stamps, and archival materials — aligned with real governmental processes and needs.

Rather than just training models on a dataset, this challenge aims to help create tools that can be adopted in practice by the Ministry of Economy, the Ministry of Digital Transformation, the State Archival Service, and many others, simplifying document processing and accelerating public service delivery.

The most successful solutions will demonstrate:

Robust handwriting recognition across different document types and writing styles;
Practical reproducibility and deployment readiness;
The ability to work with real-world, noisy, and diverse data.

Technical Constraints
Open models only — the inference pipeline must use exclusively open-weight models. Proprietary APIs (OpenAI, Anthropic, Gemini, etc.) are not permitted at inference time and for submissions!
Single H100 — the complete inference pipeline must fit into a single NVIDIA H100 80GB GPU. Solutions requiring more compute will not pass verification.

Dataset — RUKOPYS
The challenge is built around RUKOPYS — the first large-scale open dataset for Ukrainian handwritten text recognition. It covers over a century of Ukrainian handwriting across four document types:

Source	Period	Description
National Dictation	2020–2025	Phone photos of handwritten Ukrainian National Dictation submissions. Thousands of unique handwriting styles, one known canonical text per year.
State Archive (ЦДАВО)	1919–1935	Scanned documents from 12 archival funds. Pen & ink, archaic orthography.
University (KNUTE)	2022–2025	Scanned student exam work: text, formulas, chemistry, tables.
School Homework – Opornyi Lyceum s. Zymne (Опорний ліцей с. Зимне)	2022–2025	Phone photos of school homework, grades 5–11, 20+ subjects.
The dataset is publicly available on HuggingFace: UkrainianCatholicUniversity/rukopys

Train: human-annotated images (bboxes + type + transcription)
Silver: auto-annotated images for self-training
Test: submit your predictions here



Recommended Approaches
This competition is open to any method, but here are the directions we find most promising:

VLM Fine-tuning
Adapt vision-language models (Gemma 4, Qwen3-VL, LLaMA, etc.) to Ukrainian handwritten text. This is the most direct path to strong results.

Agentic Recognition Pipelines
Multi-step systems that classify document type and content, then route to specialized strategies — e.g., delegating formulas, tables, or printed text to dedicated models.

Retrieval-Augmented Recognition (RAR)
Equip your system with external knowledge bases: domain-specific vocabularies, document templates, lexical patterns. Identify the document type, retrieve relevant context, and use it to guide recognition.

Additional Data Sources
Our dataset alone may not be sufficient to train large models from scratch. We encourage creative use of external datasets, synthetic data generation, and pseudo-labeling to expand training data.

Portability
Smaller and more efficient solutions are preferred. In production, these models must run on limited hardware. Efficiency per compute unit is a competitive advantage.

Post-processing & Ensembles
Language model correction, dictionary-based fixing, and multi-model voting at the line or character level are proven ways to improve CER.



Tips & Resources
RUKOPYS is a starting point, not a ceiling. Here are proven ways to go further.

Self-training with the silver split
The silver split contains auto-annotated images — the same sources and format as train. Use it as a noisy-label pretraining stage before fine-tuning on human-verified train annotations. A simple curriculum: train on silver first, then fine-tune on gold.

Pseudo-labeling with dictation ground truth
The National Dictation ground-truth texts are publicly available for each year. Because the canonical text is known, you can skip bbox-level annotation and align the full-page transcription directly to image lines — useful for text-line-level pretraining without human annotation.

Synthetic data
Generating synthetic Ukrainian handwriting is practical and well-supported:

TextRecognitionDataGenerator (TRDG) — render text with handwriting-style fonts, distortions, custom backgrounds. Supports any language and custom glyph sets.
FbSTG — tested specifically on historical Cyrillic documents; reduced CER by 24% and WER by 8% in a published evaluation.
For Ukrainian text content there're popular open text corpora like UberText, Kobza etc. For handwriting-style fonts with full Ukrainian character coverage (Ґ, Є, І, Ї) — check Google Fonts and Font Squirrel filtering for Ukrainian support.

External datasets (permitted external data)
The following open datasets are compatible with competition rules — publicly available, non-commercial use:

Dataset	Language	Description
HKR (GitHub)	Kazakh (Cyrillic)	~63K sentences, ~200 writers, Nazarbayev University
IAM Handwriting DB	English	1,500+ pages, gold standard for Latin HTR benchmarks
Pretrained checkpoints worth fine-tuning
End-to-end document understanding (layout + OCR in one model):

Qwen/Qwen3-VL-8B-Instruct — strong vision-language model with native document understanding; fits on a single H100 with room for batching. The 72B variant is available for training if you have the compute.
Google Gemma 4 — open-weight multimodal family: 5B (gemma-4-E2B-it), 8B MoE (gemma-4-E4B-it), 27B (gemma-4-26B-A4B-it), 31B (gemma-4-31B-it). All are vision-capable and fine-tunable; the 5B–27B variants fit comfortably on a single H100.
PaddleOCR — production-grade OCR framework with pretrained detection + recognition pipelines; supports custom Cyrillic fine-tuning and has strong layout analysis tools built in.
Text-line recognition (after bbox detection):

microsoft/trocr-base-handwritten — encoder-decoder baseline for handwritten text lines; fine-tunes well on Cyrillic with modest data.
Kansallisarkisto/cyrillic-htr-model — trained on 30K+ lines of Cyrillic archival handwriting by the National Archives of Finland.
Note on Gemini: Gemini 2.5 Flash and Pro (via Google AI Studio) cannot be used in the inference pipeline, but are excellent for generating pseudo-annotations on additional unlabeled data before training your open model.

Available Сompute Resources
Training can use any hardware — only inference must fit on a single H100. Here is how to get GPU time at no cost.

AWS Credits from the organizers
Teams can apply for AWS credits through the competition organizers. Credits will be allocated during the competition period. To request — post in the competition Discord channel or contact the organizers directly.

Free notebook environments (no credit card required)
Platform	GPU	Limit	Notes
Kaggle Notebooks	P100 16GB	~30 h/week	Built into the competition — use it
Google Colab	T4 15GB	~15–30 h/week	Good for quick experiments
Lightning AI	T4	15 credits/month	100 GB persistent storage, collaborative
Amazon SageMaker Studio Lab	T4 16GB	4 h/day	Stable persistent environment
A team of 4 using Kaggle Notebooks alone gets 120+ GPU-hours/week.



Free inference APIs — useful for data preparation and annotation
Proprietary APIs cannot be used in the inference pipeline, but they are fully allowed for labeling additional data, filtering, pseudo-annotation, or any other data preparation step outside of inference.

Service	What's free
Google AI Studio	Gemini 2.5 Pro/Flash, 1M context, multimodal — excellent for OCR pre-annotation
Together AI	71+ open models free, $25 signup credits, fine-tuning supported
NVIDIA Build	1,000 free credits, 100+ open models including vision models
Groq	Free tier forever, no credit card, fast inference