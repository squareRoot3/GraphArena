# GraphArena

This repository contains the official implementation for the ICLR 2025 paper:  

> [GraphArena: Evaluating and Exploring Large Language Models on Graph Computation](openreview.net/forum?id=Y1r9yCMzeA)  
> *Jianheng Tang, Qifan Zhang, Yuhan Li, Nuo Chen, Jia Li*  
> ICLR 2025


![intro](utils/GraphArena.jpg)

## Environment Setup

```bash
conda create -n GraphArena
source activate GraphArena
conda install openai pandas numpy networkx pip
pip install pybind11
pip install rdkit ogb graph-walker
```

## Dataset Preparation

Download and unzip `dataset.zip`, which contains the processed dataset.  
To build the dataset from scratch, download `source.zip` and run `bash utils/build_dataset.sh`.

## Benchmarking LLMs

Replace `YOUR_API_KEY` in `benchmark_LLM_API.py`.

```bash
python benchmark_LLM_API.py \
  --llm {model} \
  --task {task_name} \
  --problem_num {N} \
  --example_num {K} \
  --difficulty {easy|hard} \
  --results ./results \
  --sleep 5
  --resume
```

**Key Parameters**:
- `--llm`: Model shortname (e.g., `gpt4`, `claude`, `llama8b`)
- `--task`: One of 10 graph tasks (e.g., `TSP`, `MVC`, `Diameter`)
- `--difficulty`: `easy` (small graphs) or `hard` (large graphs)
- `--problem_num`: Number of problems to evaluate (default: 500)
- `--example_num`: Number of demonstrated examples (defualt: 1).
- `--sleep`: API call cooldown (default: 5s)
- `--resume`: Resume from the last evaluation.

Details about command-line arguments are available in both `benchmark_LLM_API.py` and `utils/run_benchmark.sh`.

To evaluate LLMs locally, use:

```bash
python benchmark_LLM_local.py --llm llama8b
```

Evaluated LLMs and corresponding accuracy score:

| LLM short Name | Test Version & Date | P (small) | P (large) | NP (small) | NP (large) | Average |
| --- | --- | --- | --- | --- | --- | --- |
| dsr1 | deepseek-R1 (2025-02-15) | 0.976 | 0.877 | 0.877 | 0.431 | 0.795 |
| claude | claude-3.5-sonnet-20241022 | 0.822 | 0.587 | 0.478 | 0.072 | 0.495 |
| doubao | doubao-1.5-pro (2025-02-15) | 0.792 | 0.532 | 0.467 | 0.052 | 0.461 |
| gpt4 | gpt-4o-2024-08-06 | 0.769 | 0.435 | 0.473 | 0.063 | 0.435 |
| glm | glm-4-plus (2024-09-30) | 0.727 | 0.457 | 0.413 | 0.048 | 0.411 |
| gpt4mini | gpt-4o-mini-2024-07-18 | 0.689 | 0.366 | 0.392 | 0.033 | 0.37 |
| llama | meta-llama/Llama-3-70b-chat-hf (2024-05-30) | 0.612 | 0.316 | 0.368 | 0.047 | 0.336 |
| deepseek | deepseek-V2.5 (2024-09-30) | 0.514 | 0.247 | 0.337 | 0.031 | 0.282 |
| qwen72b | qwen2.5-72B-Instruct (2024-09-30) | 0.590 | 0.399 | 0.206 | 0.007 | 0.29 |
| llama8b | meta-llama/Llama-3-8b-chat-hf (2024-05-30) | 0.285 | 0.094 | 0.202 | 0.019 | 0.15 |
| gemma | google/gemma-1.1-7b-it (2024-05-30) | 0.252 | 0.092 | 0.129 | 0.009 | 0.12 |

For detailed metrics and analysis, see our [paper](https://openreview.net/forum?id=Y1r9yCMzeA) and [`reproduce/`](reproduce/) notebooks.

## Reproduction Guide

To reproduce the results from our manuscript, follow these steps:

1. Download and unzip both [`results.zip`](link/to/results) and `dataset.zip`
2. Run all jupyter notebooks in the `reproduce` folder

Note: The evaluation may take a few minutes to complete.

## Citation
```bibtex
@inproceedings{tang2025grapharena,
  title={GraphArena: Evaluating and Improving Large Language Models on Graph Computation},
  author={Tang, Jianheng and Zhang, Qifan and Li, Yuhan and Chen, Nuo and Li, Jia},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=Y1r9yCMzeA}
}
```