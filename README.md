AI-Model-Benchmarking-Framework

A reproducible evaluation framework for benchmarking Large Language Models (LLMs) on agent-oriented behaviors including tool-use decision making, structured output reliability, restraint handling, and runtime efficiency.

The framework is designed for researchers, AI engineers, and system developers who need standardized evaluation pipelines for comparing models across real-world agentic workloads.

Overview

As LLMs evolve into autonomous and semi-autonomous agents, traditional benchmark metrics are no longer sufficient. Modern agent systems must reliably:

Decide when external tool invocation is necessary
Select the correct tool under constrained environments
Avoid unnecessary or hallucinated actions
Produce machine-parseable structured outputs
Maintain low-latency inference characteristics

This repository provides a compact, extensible, and reproducible benchmarking harness for evaluating these behaviors across multiple model families and parameter scales.

Core Capabilities
Standardized benchmarking pipeline for agent-style LLM evaluation
Structured tool-calling assessment
Restraint and hallucination analysis
JSON validity and parsing robustness checks
Runtime performance profiling
Reproducible experiment outputs
Support for local checkpoints and Hugging Face models
Lightweight architecture for rapid experimentation
Evaluation Objectives

The framework evaluates models across both behavioral and systems-level dimensions.

Metric	Description
Action Score	Accuracy of required tool invocation
Restraint Score	Ability to avoid unnecessary tool calls
Wrong Tool Penalty	Penalization for hallucinated or incorrect actions
Valid Response Rate	Percentage of syntactically valid JSON outputs
Agent Score	Weighted composite evaluation metric
Latency	Average inference latency per request
Throughput	Tokens processed/generated per second
Benchmark Methodology

The evaluation harness uses a controlled prompt suite with a strict system prompt that:

Enumerates available tools
Enforces JSON-only responses
Restricts invalid action formats
Enables deterministic automated scoring

Prompts are divided into two benchmark categories:

1. Tool-Use Evaluation

The model must determine:

Whether a tool is required
Which tool should be selected
Whether arguments are correctly structured

Example expected output:

{
  "tool": "search_web",
  "arguments": {
    "query": "latest reinforcement learning research"
  }
}
2. Restraint Evaluation

The model must correctly avoid unnecessary external actions.

Expected output:

{
  "tool": "none"
}

This benchmark dimension is particularly important for evaluating hallucination resistance and agent reliability.

Repository Structure
AI-Model-Benchmarking-Framework/
│
├── benchmark.py
├── csv_agent.py
├── train_kcc.py
├── requirements.txt
├── report.md
│
├── results/
│   ├── *.csv
│   └── *.json
│
├── kcc_agent_checkpoints/
│
└── KCC_Call_Dataset.csv
Installation
Clone Repository
git clone https://github.com/AmanMehta22/AI-Model-Benchmarking-Framework.git
cd AI-Model-Benchmarking-Framework
Create Virtual Environment
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
Windows
python -m venv .venv
.venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Running Benchmarks

Execute the evaluation pipeline:

python benchmark.py

The framework automatically:

Loads configured models
Performs inference warmup
Executes evaluation prompts
Computes benchmark metrics
Exports structured reports
Output Artifacts

Benchmark outputs are stored in the results/ directory.

Generated Files
File	Purpose
CSV	Structured benchmark metrics
JSON	Raw evaluation outputs
report.md	Human-readable analysis and methodology

Example:

results/
├── benchmark_2026_05_20.csv
├── benchmark_2026_05_20.json
Supported Model Types

The framework supports:

Hugging Face Transformers models
Local checkpoints
Instruction-tuned chat models
Quantized inference models

Typical benchmark ranges include:

0.5B
1B
3B
7B+
Technical Design
Inference Backend

Models are loaded using:

transformers.AutoModelForCausalLM
transformers.AutoTokenizer

Additional supported capabilities:

Chat template handling
Structured prompting
GPU inference
Mixed precision execution
bfloat16 / float16 acceleration
Timing Methodology

Latency measurements are computed using:

time.perf_counter()
torch.cuda.synchronize()

This ensures accurate GPU synchronization during runtime profiling.

Engineering Design Principles

The framework prioritizes:

Reproducibility
Deterministic scoring
Minimal benchmark overhead
Extensibility
Lightweight deployment
Cross-model comparability
Representative Findings

Baseline evaluations indicate:

Larger models generally achieve higher action accuracy
Restraint handling remains a common failure mode
Structured prompting significantly improves JSON validity
Smaller models offer superior latency characteristics at reduced behavioral accuracy
Reproducibility Recommendations

For stable benchmark comparisons:

Pin CUDA and driver versions
Use fixed random seeds
Benchmark on identical hardware
Avoid concurrent GPU workloads
Maintain consistent prompt distributions
Example Evaluation Pipeline
Input Prompt
      ↓
System Prompt Injection
      ↓
Model Inference
      ↓
JSON Parsing
      ↓
Tool Decision Validation
      ↓
Metric Aggregation
      ↓
CSV / JSON Export
Use Cases

This framework is suitable for:

Agentic LLM evaluation
Tool-calling research
Structured output benchmarking
Runtime performance analysis
Fine-tuning validation
Academic experimentation
Internal model comparison pipelines
Future Roadmap

Planned enhancements include:

YAML-based configuration system
Dockerized reproducible environments
CI-integrated benchmark regression testing
Expanded adversarial prompt suites
Multi-GPU benchmarking support
Visualization dashboards
Safety and hallucination stress evaluation
Contributing

Contributions are welcome.

Recommended contribution areas:

Additional benchmark datasets
New evaluation metrics
Model integrations
Runtime optimizations
Reporting improvements
Visualization tooling

To contribute:

Fork the repository
Create a feature branch
Add reproducible benchmark outputs
Submit a Pull Request
Citation

---
# AI-Model-Benchmarking-Framework

A reproducible evaluation framework for benchmarking Large Language Models (LLMs) on agent-oriented behaviors: tool-use decision making, structured output reliability, restraint handling, and runtime efficiency. The framework targets researchers and engineers who need a standardized pipeline for comparing models on agentic workloads.

## Table of contents

- [Executive summary](#executive-summary)
- [Core capabilities](#core-capabilities)
- [Metrics & test design](#metrics--test-design)
- [Quickstart](#quickstart)
- [Repository structure](#repository-structure)
- [Running benchmarks & outputs](#running-benchmarks--outputs)
- [Technical design & timing methodology](#technical-design--timing-methodology)
- [Reproducibility recommendations](#reproducibility-recommendations)
- [Use cases & roadmap](#use-cases--roadmap)
- [Contributing](#contributing)
- [Maintainer & license](#maintainer--license)

## Executive summary

As LLMs evolve into autonomous and semi-autonomous agents, evaluation must move beyond per-token quality metrics. This framework evaluates models on agent-relevant behaviors: when to call external tools, which tool to call, whether to abstain, and whether the output is machine-parseable for safe downstream execution. It also records runtime characteristics (latency and throughput) so teams can make informed trade-offs between accuracy and cost.

Baseline runs and a full, human-readable analysis are available in [report.md](report.md). Numeric exports are saved to the `results/` directory as timestamped CSV and JSON files.

## Core capabilities

- Standardized benchmarking pipeline for agent-style evaluation
- Structured tool-calling assessment and restraint analysis
- JSON validity and parsing robustness checks
- Runtime performance profiling (latency, throughput)
- Support for Hugging Face models and local checkpoints
- Lightweight architecture for rapid experiment iteration

## Metrics & test design

- **Action Score:** accuracy on prompts that require a tool call
- **Restraint Score:** accuracy on prompts that should not call a tool
- **Valid Response Rate:** fraction of parseable JSON outputs
- **Wrong Tool Penalty:** penalizes incorrect or hallucinated tool calls
- **Latency / Throughput:** runtime profiling per-sample and tokens/sec

The harness uses a curated prompt set split into `tool_use` and `restraint` categories. A strict system prompt enumerates available tools and enforces JSON-only decisions to reduce hallucination and enable deterministic scoring.

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

3. Run the benchmark (models are configured inside `benchmark.py` by default):

```bash
python benchmark.py
```

4. Inspect outputs in the `results/` directory and read [report.md](report.md) for methodology and example findings.

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

Running `benchmark.py` performs the following steps per model:

1. Load tokenizer and model (supports HF hub and local checkpoints)
2. Warm up inference to stabilize GPU kernels
3. Execute the test suite (tool_use + restraint prompts)
4. Parse model outputs for JSON decisions
5. Aggregate metrics and write CSV/JSON reports to `results/`

Generated artifacts:

- `results/agent_benchmark_results_YYYYMMDD_HHMMSS.csv` — aggregated metrics
- `results/agent_benchmark_results_YYYYMMDD_HHMMSS.json` — detailed outputs
- `report.md` — human-readable methodology and example analysis

## Technical design & timing methodology

- Models are loaded via `transformers.AutoModelForCausalLM` and tokenizers via `AutoTokenizer`.
- Tokenization helpers support tokenizers that expose `apply_chat_template` with fallbacks for broader compatibility.
- Timing uses `time.perf_counter()` with `torch.cuda.synchronize()` before and after generation to obtain accurate GPU timings.

## Reproducibility recommendations

- Pin CUDA, driver, and PyTorch versions where possible.
- Use fixed random seeds (the harness contains a seed constant).
- Run benchmarks on identical hardware configurations to compare latency and throughput reliably.

## Use cases & roadmap

Use cases:

- Agentic LLM evaluation
- Tool-calling research
- Structured output benchmarking
- Runtime performance analysis and fine-tuning validation

Planned enhancements:

- YAML-based configuration
- Dockerized environments for parity
- CI-driven regression checks on a small prompt set
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

If you want further tweaks (add badges, CI examples, a Dockerfile, or direct links to the latest `results/` files), tell me which item to add and I will implement it.