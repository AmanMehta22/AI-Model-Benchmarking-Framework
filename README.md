# KCC Agent — Knowledge-Centered Chat Benchmark

A compact research and experiment repository for training, evaluating, and benchmarking a small conversational agent focused on the KCC dataset. This project contains training scripts, saved adapter-style checkpoints, and evaluation/benchmark utilities used to compare agent variants.

**Project Goals**
- **Reproducibility:** provide clear scripts and checkpoints to reproduce training runs.
- **Lightweight experiments:** use adapter-style checkpoints for fast iteration and smaller artifacts.
- **Reliable evaluation:** include benchmarking and result logging to compare models quantitatively.

**Repository Structure**
- **`train_kcc.py`**: Training script and configuration entrypoint.
- **`csv_agent.py`**: Lightweight agent wrapper for dataset interaction and inference.
- **`benchmark.py`**: Scripts to run evaluation benchmarks and produce CSV/JSON results.
- **`KCC_Call_Dataset.csv`**: Primary dataset used for training and evaluation.
- **`kcc_agent_checkpoints/`**: Saved checkpoints (adapter and tokenizer artifacts).
- **`results/`**: Generated benchmark outputs, including timestamps for reproducibility.

Quickstart

1. Create a Python virtual environment and install dependencies (example using pip):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train (or resume) a model using the provided training script:

```bash
python train_kcc.py --config configs/train_config.yaml
```

3. Run a benchmark using the trained checkpoint(s):

```bash
python benchmark.py --checkpoint kcc_agent_checkpoints/checkpoint-300
```

Notes on Checkpoints and Templates
- Checkpoints are stored under `kcc_agent_checkpoints/` with a small README inside each checkpoint directory describing its contents.
- Each checkpoint includes an adapter model file and tokenizer config. Loading typically follows the pattern used in `csv_agent.py`.

Results and Reporting
- Benchmark outputs are written to `results/` as CSV and JSON files. See `results/agent_benchmark_results_20260520_091620.csv` for an example.
- Keep benchmark runs deterministic where possible by setting RNG seeds in training and evaluation.

Contributing
- Open an issue to propose changes or discuss experiment ideas.
- For PRs: include a short description, training configuration, and a small evaluation that demonstrates any changes.

License & Contact
- This repository is provided for research and experimentation. If you plan to use or adapt the code for commercial purposes, contact the maintainer for licensing details.
- For questions or collaboration, open an issue or email the author listed in the project metadata.

---