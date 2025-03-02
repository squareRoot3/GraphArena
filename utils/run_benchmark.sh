# script for benchmarking LLMs on GraphArena through API calling. 10 tasks will be run in order.

problem_num=500  # number of problems to be benchmarked in each setting
example_num=1    # number of examples in each problem
difficulty="hard"  # difficulty level of the problems: easy (small graphs) or hard (large graphs)
results="p_hard"    # folder to save the results
llm='dsR1'        # LLM to be benchmarked
resume=1        # resume the benchmarking process if the results already exist
sleep=0        # sleep time between two consecutive API calls

python benchmark_LLM_API.py --task Neighbor --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/Neighbor.log &
python benchmark_LLM_API.py --task Diameter --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/Diameter.log &
python benchmark_LLM_API.py --task Distance --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/Distance.log &
python benchmark_LLM_API.py --task Connected --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/Connected.log &
python benchmark_LLM_API.py --task GED --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/GED.log &
python benchmark_LLM_API.py --task MCP --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/MCP.log &
python benchmark_LLM_API.py --task MCS --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/MCS.log &
python benchmark_LLM_API.py --task MIS --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/MIS.log &
python benchmark_LLM_API.py --task MVC --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/MVC.log &
python benchmark_LLM_API.py --task TSP --problem_num $problem_num --example_num $example_num --results $results --llm $llm --difficulty $difficulty --resume $resume --sleep $sleep > log/TSP.log &
