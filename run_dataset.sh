# script for building the GraphArena problem set. 10 tasks will be processed in parallel.
loc="dataset"
mkdir -p "logs"
python build_dataset.py --task TSP --loc $loc > logs/tsp.log &
python build_dataset.py --task GED --loc $loc > logs/ged.log &
python build_dataset.py --task MCS --loc $loc > logs/mcs.log &
python build_dataset.py --task MCP --loc $loc > logs/mcp.log &
python build_dataset.py --task MVC --loc $loc > logs/mvc.log &
python build_dataset.py --task MIS --loc $loc > logs/mis.log &
python build_dataset.py --task Distance --loc $loc > logs/Distance.log &
python build_dataset.py --task Diameter --loc $loc > logs/Diameter.log &
python build_dataset.py --task Connected --loc $loc > logs/Connected.log &
python build_dataset.py --task Neighbor --loc $loc > logs/Neighbor.log &