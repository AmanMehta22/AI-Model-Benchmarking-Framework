git clone https://github.com/AmanMehta22/AI-Model-Benchmarking-Framework.git
---
# AI Model Benchmarking Framework

A reproducible evaluation framework for benchmarking Large Language Models (LLMs) on agent-oriented behaviors, including tool-use decision making, structured output reliability, restraint handling, and runtime efficiency. This repository provides a professional and extensible harness for engineers and researchers to compare models on agentic workloads.

## Table of contents

- [Overview](#overview)
- [Capabilities](#capabilities)
- [Metrics](#metrics)
- [Methodology](#methodology)
- [Quickstart](#quickstart)
- [Repository structure](#repository-structure)
- [Running benchmarks & outputs](#running-benchmarks--outputs)
- [Technical details & timing methodology](#technical-details--timing-methodology)
- [Reproducibility recommendations](#reproducibility-recommendations)
- [Use cases & roadmap](#use-cases--roadmap)
- [Contributing](#contributing)
- [Maintainer & license](#maintainer--license)

## Overview

As LLMs are integrated into agentic systems, evaluation must address behavioral correctness and runtime characteristics. This framework focuses on three behavioral axes (action, restraint, validity) and system performance (latency, throughput), enabling repeatable comparisons across model families and sizes.

Detailed methodology and baseline findings are provided in `report.md`. Numeric exports are written to the `results/` directory as timestamped CSV and JSON files.

## Capabilities

- Standardized benchmarking pipeline for agent-style evaluation
- Tool-use decision assessment and restraint analysis
- JSON validity checks and robust parsing
- Latency and throughput profiling
- Support for Hugging Face models and local checkpoints
- Lightweight design for rapid iteration and extension

## Metrics

| Metric | Description |
|---|---|
| Action Score | Accuracy on prompts that require a tool call |
| Restraint Score | Accuracy on prompts where no tool should be called |
| Wrong Tool Calls | Count of incorrect or hallucinated tool selections |
| Valid Response Rate | Fraction of outputs that are parseable JSON |
| Agent Score | Weighted composite score combining the above metrics |
| Latency | Average inference time per prompt (ms) |
| Throughput | Tokens generated per second |

## Methodology

The benchmark uses a curated prompt suite and a strict system prompt that enumerates available tools and enforces JSON-only responses. Prompts are categorized into `tool_use` (requires tool invocation) and `restraint` (should return `{"tool": "none"}`). Outputs are parsed and validated; metrics are aggregated per model and exported.

## Quickstart

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the benchmark (models are configured in `benchmark.py`):

```bash
python benchmark.py
```

4. Inspect results in the `results/` directory and consult `report.md` for methodology and example findings.

## Repository structure

```
AI-Model-Benchmarking-Framework/
├── benchmark.py
├── csv_agent.py
├── train_kcc.py
├── requirements.txt
├── report.md
├── results/            # CSV and JSON exports (timestamped)
├── kcc_agent_checkpoints/
└── KCC_Call_Dataset.csv
```

## Running benchmarks & outputs

Running `benchmark.py` performs the following steps for each model:

1. Load tokenizer and model (Hugging Face hub or local checkpoint)
2. Warm up inference to stabilize GPU kernels
3. Execute the test suite (tool_use + restraint prompts)
4. Parse model outputs for JSON decisions
5. Aggregate metrics and write CSV/JSON reports to `results/`

Generated artifacts:

- `results/agent_benchmark_results_YYYYMMDD_HHMMSS.csv` — aggregated metrics
- `results/agent_benchmark_results_YYYYMMDD_HHMMSS.json` — detailed model outputs
- `report.md` — human-readable methodology and example analysis

## Technical details & timing methodology

- Models are loaded via `transformers.AutoModelForCausalLM` and tokenizers via `AutoTokenizer`.
- Tokenization helpers include support for `apply_chat_template` with fallbacks for compatibility.
- Timing uses `time.perf_counter()` with `torch.cuda.synchronize()` before and after generation to provide accurate GPU timings.

## Reproducibility recommendations

- Pin CUDA, driver, and PyTorch versions when possible.
- Use fixed random seeds (the harness contains a seed constant).
- Run benchmarks on identical hardware configurations to compare latency and throughput reliably.

## Use cases & roadmap

Use cases:

- Agentic LLM evaluation
- Tool-calling research and validation
- Structured-output benchmarking
- Runtime performance analysis and fine-tuning validation

Planned enhancements:

- YAML-based configuration
- Dockerized environments for parity
- CI-driven benchmark regression checks
- Expanded adversarial and safety prompt suites

## Contributing

Contributions are welcome. Recommended workflow:

1. Fork the repository
2. Create a feature branch
3. Add reproducible benchmark outputs and tests
4. Submit a Pull Request with a short description of changes

## Maintainer & license

Maintainer: Aman Kumar Mehta

License: MIT License © 2026 Aman Kumar Mehta

---