# GraphArena Benchmark

This is the official implementation of the following manuscript:

>GraphArena: Benchmarking Large Language Models on Graph Computational Problems  
*Jianheng Tang, Qifan Zhang, Yuhan Li, Jia Li*  
NeurIPS 2023 Datasets and Benchmarks Track Submission  

## Environment Setup

To set up the environment, follow these steps:
```bash
conda create -n GraphArena
source activate GraphArena
conda install openai pandas numpy networkx matplotlib pip
pip install rdkit ogb pybind11 graph-walker
```

## Dataset Preparation

You can download and unzip the processed dataset `dataset.zip` directly from [Google Drive](https://drive.google.com/drive/folders/1mvJSUTrfOX13wgpkyb3w8s_SJqipnb1c?usp=sharing).

Alternatively, you can construct the dataset from the source data. Download and unzip `source.zip`, then run `run_dataset.sh`. Note that the constructed dataset may differ due to randomness in sampling.

## LLM Inference

To run LLM inference, use the following command:

```bash
python benchmark_LLM.py --task $task --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
```

For example, to run GPT on the full list (500 problems) on the small graphs of the TSP task:

```bash
python benchmark_LLM.py --task TSP --problem_num 500 --llm gpt --difficulty easy
```

To benchmark an LLM on all tasks, refer to `run_benchmark.sh`. The meaning of all arguments can also be found in `benchmark_LLM.py` and `run_benchmark.sh`.

The following LLM models are supported:

```json
"gpt4": "gpt-4o",
"gpt": "gpt-3.5-turbo-0125", 
"claude": "claude-3-haiku-20240307",
"mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
"deepseek": "deepseek-chat",
"llama8b": "meta-llama/Llama-3-8b-chat-hf",
"llama": "meta-llama/Llama-3-70b-chat-hf",
"qwen7b": "qwen1.5-7b-chat",
"qwen": "qwen1.5-72b-chat",
"gemma": "gemma-7b-it",
```

## LLM Evaluation

Unzip `final_results.zip` and run `score_LLM.ipynb` to reproduce the results in the paper.

### Case Demonstration

We have integrated the problem text and all results into a single JSON file `GraphArena_all.json` for reference. The file is organized in the following format:

```json
{
    "Task_name": [
        {
            "id": 0,  // 0-499 for small graphs (easy) and 500-999 for large graphs (hard)
            "problem_text": "...",
            "LLM responses": "..."
        },
        ...
    ]
    ...
}
```

More examples can be found in `GraphArena_all.json` and `GraphArena_example.txt`.

## License

The dataset is under the CC BY-SA 4.0 License. The code repository is under the BSD-2 License.
