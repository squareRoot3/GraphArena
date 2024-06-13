# script for benchmark LLMs on GraphArena through API calling. 10 tasks will be run in order.

problem_num=500  # number of problems to be benchmarked in each setting
example_num=1    # number of examples in each problem
difficulty="easy"  # difficulty level of the problems: easy (small graphs) or hard (large graphs)
results="p_easy"    # folder to save the results
llm='llama3'        # LLM to be benchmarked
resume=1        # resume the benchmarking process if the results already exist
sleep=0        # sleep time between two consecutive API calls

python benchmark_LLM.py --task Neighbor --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
python benchmark_LLM.py --task Diameter --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
python benchmark_LLM.py --task Distance --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
python benchmark_LLM.py --task Connected --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep 
python benchmark_LLM.py --task GED --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
python benchmark_LLM.py --task MCP --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
python benchmark_LLM.py --task MCS --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
python benchmark_LLM.py --task MIS --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
python benchmark_LLM.py --task MVC --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
python benchmark_LLM.py --task TSP --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep
