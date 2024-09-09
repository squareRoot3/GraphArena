# script for benchmarking three GNNs on all 10 tasks in GraphArena.
python benchmark_GNN.py --task Distance --model GIN
python benchmark_GNN.py --task Distance --model GAT
python benchmark_GNN.py --task Distance --model SAGE
python benchmark_GNN.py --task Neighbor --model GIN
python benchmark_GNN.py --task Neighbor --model GAT
python benchmark_GNN.py --task Neighbor --model SAGE
python benchmark_GNN.py --task Diameter --model GIN
python benchmark_GNN.py --task Diameter --model GAT
python benchmark_GNN.py --task Diameter --model SAGE
python benchmark_GNN.py --task Connected --model GIN
python benchmark_GNN.py --task Connected --model GAT
python benchmark_GNN.py --task Connected --model SAGE
python benchmark_GNN.py --task MVC --model GIN
python benchmark_GNN.py --task MVC --model GAT
python benchmark_GNN.py --task MVC --model SAGE
python benchmark_GNN.py --task MIS --model GIN
python benchmark_GNN.py --task MIS --model GAT
python benchmark_GNN.py --task MIS --model SAGE
python benchmark_GNN.py --task MCP --model GIN
python benchmark_GNN.py --task MCP --model GAT
python benchmark_GNN.py --task MCP --model SAGE
python benchmark_GNN.py --task GED --model GIN
python benchmark_GNN.py --task GED --model GAT
python benchmark_GNN.py --task GED --model SAGE
python benchmark_GNN.py --task MCS --model GIN
python benchmark_GNN.py --task MCS --model GAT
python benchmark_GNN.py --task MCS --model SAGE
python benchmark_GNN.py --task TSP --model GIN
python benchmark_GNN.py --task TSP --model GAT
python benchmark_GNN.py --task TSP --model SAGE