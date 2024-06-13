# list all files name in the results dir
import os
import json
from collections import defaultdict
from tasks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
merged_responses = {}
problem_num = 500
dataset_loc = 'dataset'


paths = ['final_results/results_for_plot/radar/easy', 'final_results/results_for_plot/radar/hard']

all_files = {}

for path in paths:
    difficulty = path.split('/')[-1]
    files = os.listdir(path)
    all_files[difficulty] = files
    # all_files[difficulty]['path']=path
    
print(all_files)
# print(all_files.items())
    
for difficulty, files in all_files.items():
    for f in files:
        if len(f.split('_')) < 2:
            continue
        llm, task = f.split('_')[0], f.split('_')[1]
        
        path = 'final_results/results_for_plot/radar/' + difficulty
        
        with open(f'{path}/{f}', 'r') as file:
            response_dict = json.load(file)
            
        for i in range(0, problem_num):
            if task not in merged_responses:
                merged_responses[task] = defaultdict(dict)
            
            if difficulty not in merged_responses[task]:
                merged_responses[task][difficulty] = defaultdict(dict)
            merged_responses[task][difficulty][i][llm] = response_dict[str(i)][llm]
            
task_list = list(merged_responses.keys())
print(task_list)


score = {}
for task_name in task_list:
    task= globals()[task_name + '_Task'](dataset_loc)
    difficulty_level = ['easy','hard']
    score[task_name] = defaultdict(dict)
    for difficulty in difficulty_level:
        task.load_dataset(difficulty)
        score[task_name][difficulty] = defaultdict(dict)
        for i in range(0, problem_num):
            score[task_name][difficulty][i]['gt'] = task.problem_set[i]['exact_answer']
            for llm in merged_responses[task_name][difficulty][i].keys():
                if llm == 'problem':
                    continue
                r = merged_responses[task_name][difficulty][i][llm]
                if r is None:
                    r = ''
                    print(i, llm, task_name)
                # print(r)
                # r = r.replace('\\text{', '').replace('}', '').replace('\[', '').replace('\]', '')
                score[task_name][difficulty][i][llm] = task.check_solution(i, r)
# json.dump(score, open('score.json', 'w'))


metrics = defaultdict(dict)
less_is_better = ['GED', 'TSP', 'MVC', 'Distance']
results = []
cnt=0
for task in task_list:
    print(task)
    model_list = list(score[task]['easy'][0].keys())
    model_list.remove('gt')
    for model in model_list:
        metrics[task][model] = {'feasible-easy':[], 'acc-easy': [], 'feasible-hard':[],'acc-hard':[]}
        for i in range(0, problem_num):
            metrics[task][model]['feasible-easy'].append(score[task]['easy'][i][model]>=0)
            metrics[task][model]['acc-easy'].append(score[task]['easy'][i][model]==score[task]['easy'][i]['gt'])
            
            metrics[task][model]['feasible-hard'].append(score[task]['hard'][i][model]>=0)
            metrics[task][model]['acc-hard'].append(score[task]['hard'][i][model]==score[task]['hard'][i]['gt'])
            
        avg_feasible_easy = sum(metrics[task][model]['feasible-easy']) / problem_num
        avg_acc_easy = sum(metrics[task][model]['acc-easy']) / problem_num

        avg_feasible_hard = sum(metrics[task][model]['feasible-hard']) / problem_num
        avg_acc_hard = sum(metrics[task][model]['acc-hard']) / problem_num

        results.append((task, model, avg_feasible_easy,  avg_acc_easy, avg_feasible_hard, avg_acc_hard))
        cnt = cnt +1
        print(cnt, len(results))
        

# Sorting the results by MRR for each task
sorted_results = defaultdict(list)

for task in task_list:
    task_results = [result for result in results if result[0] == task]
    sorted_results[task] = sorted(task_results, key=lambda x: x[4], reverse=True)  # Sort by MRR

# Print sorted results for each task
for task, task_results in sorted_results.items():
    print(f"\nTask: {task}")
    for result in task_results:
        print(f"Model: {result[1]}, avg_feasible_easy: {result[2]:.3f},  avg_acc_easy: {result[3]:.3f}, avg_feasible_hard: {result[4]:.3f},avg_acc_hard: {result[5]:.3f}")

# Calculate average MRR performance across all tasks for each model
model_metrics = defaultdict(lambda: defaultdict(list))

# Aggregate metrics for each model across tasks
for result in results:
    task, model, avg_feasible_easy,avg_acc_easy, avg_feasible_hard, avg_acc_hard = result
    model_metrics[model]['feasible-easy'].append(avg_feasible_easy)
    model_metrics[model]['acc-easy'].append(avg_acc_easy)
    model_metrics[model]['feasible-hard'].append(avg_feasible_hard)
    model_metrics[model]['acc-hard'].append(avg_acc_hard)

# Compute average metrics for each model
average_metrics_performance = {model: {metric: sum(values) / len(values) for metric, values in metrics.items()} for model, metrics in model_metrics.items()}

# Sort models by their average MRR
sorted_average_metrics = sorted(average_metrics_performance.items(), key=lambda x: x[1]['acc-easy'], reverse=True)

# Print the sorted average metrics for each model
print("\nAverage Performance Across All Tasks:")
for model, metrics in sorted_average_metrics:
    print(f"Model: {model}, Average feasible-easy: {metrics['feasible-easy']:.3f}, Average acc-easy: {metrics['acc-easy']:.3f}, Average feasible-hard: {metrics['feasible-hard']:.3f}, Average acc-hard: {metrics['acc-hard']:.3f}")
    


print(results)

task_order = ['Neighbor', 'Distance', 'Connected', 'Diameter', 'MCP', 'GED', 'MCS', 'MIS', 'MVC', 'TSP']

columns = ['Task', 'Model', 'feasible-easy', 'acc-easy', 'feasible-hard', 'acc-hard']
df = pd.DataFrame(results, columns=columns)
print(df.keys())

print(df)

df.to_excel('results.xlsx', index=False)


df = pd.read_excel('results.xlsx')

df = df.sort_values(by='Model')

print(df['Model'])

#task_order = ['Neighbor', 'Distance', 'Component', 'Diameter', 'MCP', 'GED', 'MCS', 'MIS', 'MVC', 'TSP']
columns = ['Task', 'Model', 'feasible-easy', 'acc-easy', 'feasible-hard', 'acc-hard']

plt.rcParams['font.family'] = 'Times New Roman'
average_metrics = defaultdict(lambda: defaultdict(dict))

for model in df['Model'].unique():
    if model not in ['gpt4','deepseek','claude','llama','mixtral']:
        continue

    
    for task in df['Task'].unique():
        task_data = df[(df['Model'] == model) & (df['Task'] == task)]
        if not task_data.empty:
            average_metrics[model][task] = {
                'Feasibility on Small Graphs': task_data['feasible-easy'].mean(),
                'Accuracy on Small Graphs': task_data['acc-easy'].mean(),
                'Feasibility on Large Graphs': task_data['feasible-hard'].mean(),
                'Accuracy on Large Graphs': task_data['acc-hard'].mean()
            }


def reorder_metrics(metrics, order):
    reordered = {}
    for task in order:
        if task in metrics:
            reordered[task] = metrics[task]
    return reordered


def plot_and_save_radar_chart_for_all_metrics(filename):
    tasks = task_order
    N = len(tasks)


    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(polar=True))
    axes = axes.flatten()  

    for i, ax in enumerate(axes):
        metric_name = metrics_to_plot[i]
        for model in average_metrics.keys():
            metrics = reorder_metrics(average_metrics[model], task_order)
            values = [metrics[task][metric_name] for task in tasks]
            values += values[:1]  
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1) 

        ax.set_ylim(0, 1.0)
        
        ax.spines['polar'].set_color('gray')
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        
    
        ax.set_xticklabels(tasks, fontsize=25) 
        ax.set_title(f'{metric_name}', size=30, y=1.05) 

    handles, labels = axes[0].get_legend_handles_labels()
    labels = ['Claude3-haiku','Deepseek-V2','GPT-4o','Llama3-70b','Mixtral-7x8b']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels), fancybox=True, fontsize=25)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(filename, format='pdf', dpi=1200, bbox_inches='tight')  
    plt.show()

metrics_to_plot = ['Feasibility on Small Graphs', 'Feasibility on Large Graphs', 'Accuracy on Small Graphs', 'Accuracy on Large Graphs']
filename = 'figure2_radar.pdf'
plot_and_save_radar_chart_for_all_metrics(filename)