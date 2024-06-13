
import os
import json
from collections import defaultdict
from tasks import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
merged_responses = {}
problem_num = 500
dataset_loc = 'dataset'

paths = ['final_results/results_for_plot/cot/63_0shot', 'final_results/results_for_plot/cot/63_1shot','final_results/results_for_plot/cot/63_2shot','final_results/results_for_plot/cot/63_3shot','final_results/results_for_plot/cot/63_4shot']

all_files = {}

for path in paths:
    shot_num = path.split('_')[-1]
    files = os.listdir(path)
    all_files[shot_num] = files
    # all_files[difficulty]['path']=path
    
print(all_files['3shot'])
# print(all_files.items())
    
for shot_num, files in all_files.items():
    for f in files:
        if len(f.split('_')) < 2:
            continue
        llm, task = f.split('_')[0], f.split('_')[1]
        # if llm in ['gpt4','deepseek']:
        #     continue
        path = 'final_results/results_for_plot/cot/63_' + shot_num
        
        with open(f'{path}/{f}', 'r') as file:
            response_dict = json.load(file)
            
        for i in range(0, problem_num):
            if task not in merged_responses:
                merged_responses[task] = defaultdict(dict)
            
            if shot_num not in merged_responses[task]:
                merged_responses[task][shot_num] = defaultdict(dict)
            merged_responses[task][shot_num][i][llm] = response_dict[str(i)][llm]
            
task_list = list(merged_responses.keys())
print(task_list)


score = {}

for task_name in task_list:
    task= globals()[task_name + '_Task'](dataset_loc)
    shot_level = ['0shot','1shot','2shot','3shot','4shot']
    score[task_name] = defaultdict(dict)
    for shot in shot_level:
        task.load_dataset('easy')
        score[task_name][shot] = defaultdict(dict)
        for i in range(0, problem_num):
            score[task_name][shot][i]['gt'] = task.problem_set[i]['exact_answer']
            for llm in merged_responses[task_name][shot][i].keys():
                if llm == 'problem':
                    continue
                r = merged_responses[task_name][shot][i][llm]
                if r is None:
                    r = ''
                    print(i, llm, task_name)
                # print(r)
                # r = r.replace('\\text{', '').replace('}', '').replace('\[', '').replace('\]', '')
                score[task_name][shot][i][llm] = task.check_solution(i, r)


metrics = defaultdict(dict)
less_is_better = ['Distance']
results = []
cnt=0
for task in task_list:
    print(task)
    model_list = list(score[task]['0shot'][0].keys())
    model_list.remove('gt')
    for model in model_list:
        metrics[task][model] = {'feasibility-0shot':[], 'accuracy-0shot': [], 'feasibility-1shot':[], 'accuracy-1shot': [], 'feasibility-2shot':[], 'accuracy-2shot': [],'feasibility-3shot':[], 'accuracy-3shot': [], 'feasibility-4shot':[], 'accuracy-4shot': []}
        for i in range(0, problem_num):
            metrics[task][model]['feasibility-0shot'].append(score[task]['0shot'][i][model]>=0)
            metrics[task][model]['accuracy-0shot'].append(score[task]['0shot'][i][model]==score[task]['0shot'][i]['gt'])
            
            metrics[task][model]['feasibility-1shot'].append(score[task]['1shot'][i][model]>=0)
            metrics[task][model]['accuracy-1shot'].append(score[task]['1shot'][i][model]==score[task]['1shot'][i]['gt'])
            
            metrics[task][model]['feasibility-2shot'].append(score[task]['2shot'][i][model]>=0)
            metrics[task][model]['accuracy-2shot'].append(score[task]['2shot'][i][model]==score[task]['2shot'][i]['gt'])
            
            metrics[task][model]['feasibility-3shot'].append(score[task]['3shot'][i][model]>=0)
            metrics[task][model]['accuracy-3shot'].append(score[task]['3shot'][i][model]==score[task]['3shot'][i]['gt'])
            
            metrics[task][model]['feasibility-4shot'].append(score[task]['4shot'][i][model]>=0)
            metrics[task][model]['accuracy-4shot'].append(score[task]['4shot'][i][model]==score[task]['4shot'][i]['gt'])
        

        avg_feasible_0shot = sum(metrics[task][model]['feasibility-0shot']) / problem_num
        avg_acc_0shot = sum(metrics[task][model]['accuracy-0shot']) / problem_num

        avg_feasible_1shot = sum(metrics[task][model]['feasibility-1shot']) / problem_num
        avg_acc_1shot = sum(metrics[task][model]['accuracy-1shot']) / problem_num
        
        avg_feasible_2shot = sum(metrics[task][model]['feasibility-2shot']) / problem_num
        avg_acc_2shot = sum(metrics[task][model]['accuracy-2shot']) / problem_num
        
        avg_feasible_3shot = sum(metrics[task][model]['feasibility-3shot']) / problem_num
        avg_acc_3shot = sum(metrics[task][model]['accuracy-3shot']) / problem_num
        
        avg_feasible_4shot = sum(metrics[task][model]['feasibility-4shot']) / problem_num
        avg_acc_4shot = sum(metrics[task][model]['accuracy-4shot']) / problem_num
        
        

        results.append((task, model, avg_feasible_0shot,  avg_acc_0shot, avg_feasible_1shot, avg_acc_1shot, avg_feasible_2shot, avg_acc_2shot , avg_feasible_3shot, avg_acc_3shot, avg_feasible_4shot, avg_acc_4shot))
        cnt = cnt +1
        print(cnt, len(results))
print(results)

# with open('results1.pkl', 'wb') as f:
    # pickle.dump(results, f)
    

plt.rcParams.update({'font.size': 30, 'font.family': 'Times New Roman'})
tasks = list(set([result[0] for result in results]))
models = list(set([result[1] for result in results]))

models = sorted(models)

def plot_task_performance(results, task, ax_feasibility, ax_accuracy):
    shots = [0, 1, 2, 3, 4]
    
    for model in models:
        if model not in ['gpt4','deepseek','claude','llama','mixtral']:
            continue
        task_results = [result for result in results if result[0] == task and result[1] == model]
        if task_results:
            task_result = task_results[0]
            feasible = task_result[2:11:2]
            acc = task_result[3:12:2]
            ax_feasibility.plot(shots, feasible, marker='o', label=model)
            ax_accuracy.plot(shots, acc, marker='o', label=model)
    
    task1 = task
    if task1 == 'Connected':
        task1 = 'Component'
        
    ax_feasibility.set_title(f'Feasibility: {task1} ', fontsize=30, fontname='Times New Roman')
    ax_feasibility.set_xlabel('Shots', fontsize=30, fontname='Times New Roman')
    #ax_feasibility.set_ylabel('Feasibility', fontsize=25, fontname='Times New Roman')
    ax_feasibility.set_xticks(shots)
    ax_feasibility.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax_feasibility.tick_params(axis='both', which='major', labelsize=20)
    
    ax_accuracy.set_title(f'Accuracy: {task1} ', fontsize=30, fontname='Times New Roman')
    ax_accuracy.set_xlabel('Shots', fontsize=30, fontname='Times New Roman')
    #ax_accuracy.set_ylabel('Accuracy', fontsize=25, fontname='Times New Roman')
    ax_accuracy.set_xticks(shots)
    ax_accuracy.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax_accuracy.tick_params(axis='both', which='major', labelsize=20)


fig, axs = plt.subplots(1, len(tasks)*2, figsize=(18, 6))

for i, task in enumerate(tasks):
    plot_task_performance(results, task, axs[i*2], axs[i*2+1])


plt.tight_layout()


handles, labels = axs[0].get_legend_handles_labels()
print(labels)
labels = ['Claude3-haiku','Deepseek-V2','GPT-4o','Llama3-70b','Mixtral-7x8b']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(labels), fontsize=25)

plt.subplots_adjust(top=0.8)
plt.savefig('figure4_cot.pdf', format='pdf', dpi=1200, bbox_inches='tight')  

plt.show()
