````md
## AI Model Benchmarking Framework

This repository contains a small benchmarking and fine-tuning setup for Kisan Call Center style agricultural Q&A and agent-style tool-use evaluation.

## What the code does

- `benchmark.py` evaluates a set of Hugging Face instruction models on a balanced 36-prompt suite that tests tool-use decisions, restraint, JSON validity, latency, and throughput.
- `train_kcc.py` prepares a QLoRA-style fine-tuning run for `Qwen/Qwen2.5-1.5B-Instruct` using the local CSV dataset.
- `csv_agent.py` sends the full CSV content to a local Ollama server at `http://localhost:11434/api/generate` with `num_ctx = 32768`.

## Known implementation details

- `benchmark.py` uses `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`, `Mistral-7B`, and `LFM-2.5-1.2B`.
- The benchmark results in `results/agent_benchmark_results_20260520_091620.csv` match the values described in `report.md`.
- `csv_agent.py` currently expects CSV columns named `question` and `answer`, but `KCC_Call_Dataset.csv` uses `questions` and `answers`. That mismatch needs a code fix for the script to run successfully.
- `train_kcc.py` expects `questions` and `answers`, which matches the CSV headers.

## Accuracy notes

The previous version of this file overstated a few things. The code does not guarantee factual recall, hallucination-free answers, or production-grade deployment by itself. It is better described as a local benchmarking and training prototype with an Ollama-backed CSV agent.

## Practical summary

- The benchmark is real and reproducible from the checked-in results.
- The fine-tuning pipeline is present, but it is still a training script rather than a complete deployment workflow.
- The CSV agent is the most fragile part because of the column-name mismatch.
````
