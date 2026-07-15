# ProRead AI Demo

ProRead AI is a text simplification tool built to help dyslexic readers get through dense academic writing without the cognitive overload. It takes complex passages and rewrites them in plain language, preserving the original meaning while making them much easier to read. The demo runs on Gradio and is deployed on Hugging Face Spaces.

## Overview

The project uses a fine-tuned T5 transformer model to simplify text through a sequence-to-sequence pipeline. You paste in complex text, hit the button, and get back a simplified version with readability metrics showing exactly how much the grade level dropped.

## Purpose

Academic writing is hard to read because of its long sentences, multi-syllabic jargon, dense formatting. For dyslexic readers, that cognitive load can make comprehension nearly impossible. ProRead AI tackles that in two ways:

1. **Linguistic Simplification:** Lowers the Flesch-Kincaid Grade Level by reducing average sentence length and syllables per word.
2. **Visual Design:** Uses dyslexia-friendly typography (OpenDyslexic font, increased line height and letter spacing) to reduce visual stress alongside the simplified text.

## Features

* **Interactive UI:** Built with Gradio
* **Fine-Tuned Simplification Model:** T5 model fine-tuned on WikiLarge using a `simplify:` task prefix to guide the transformation.
* **Beam Search Decoding:** Uses 10-beam search, temperature 0.9, and a no-repeat n-gram constraint to keep outputs non-repetitive.
* **Readability Metrics:** Displays Flesch-Kincaid grade level, average sentence length, and average syllables per word before and after simplification.
* **Hugging Face Deployment:** Configured to run directly on HF Spaces.

## Technologies Used

* **Language:** Python
* **ML Framework:** Hugging Face Transformers & Datasets
* **Model:** T5 (Text-to-Text Transfer Transformer) — `T5-Small`
* **Interface:** Gradio (v6.13.0)
* **Training Data:** WikiLarge Dataset (`eilamc14/wikilarge-clean`)

## Project Structure

```text
├── proread_ai_final_model/  # fine-tuned model weights and tokenizer
├── .gitattributes          # git LFS configuration
├── README.md               
├── app.py                  # gradio app and simplification pipeline
└── requirements.txt        # dependencies
```
