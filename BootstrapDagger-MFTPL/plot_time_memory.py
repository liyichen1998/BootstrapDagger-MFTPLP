import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# env_list = ['HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HopperBulletEnv-v0']
env_list = ['AntBulletEnv-v0', 'HopperBulletEnv-v0', 'HalfCheetahBulletEnv-v0','Walker2DBulletEnv-v0']

name_ids = ['bc_1', 'dagger_1', 'logger_1', 'logger_5', 'logger_25', 'mftpl_25']

label_ids = ['BC','DAgger','BD-1','BD-5','BD-25','MP-25(15)']

name_ids_for_ablation_study = ['expert', 'dagger_1',  'logger_5', 'dagger_1_running_on_logger_5', 'logger_5_running_on_dagger_1']#  , 'logger_25'

name_ids_for_ablation_study_2 = ['expert', 'dagger_1',  'logger_5',  'logger_25', 'dagger_5', 'dagger_25', 'elogger_5'] #, 'mdagger_5'
    
name_ids_for_mix_vs_mean = ['logger_5', 'logger_25']

name_ids_no_expert = ['bc_1', 'dagger_1', 'logger_1', 'logger_5', 'logger_25']

# name_mapping_dict = {
# 'expert': 'Expert',    
# 'bc_1': 'Behavior Cloning',
# 'dagger_1': 'DAgger',
# 'logger_1': 'B-DAgger-1',
# 'logger_5': 'B-DAgger-5',
# 'logger_25': 'B-DAgger-25',
# 'dagger_1_running_on_logger_5': 'DAgger on Trails of B-DAgger-5',
# 'logger_5_running_on_dagger_1': 'B-DAgger-5 on Trails of DAgger'
# }

name_mapping_dict = {
'expert': 'Expert',    
'bc_1': 'BC',
'dagger_1': 'DAgger',
'dagger_5': 'DAgger-5',
'dagger_25': 'DAgger-25',
'mftpl_25': 'MP-25(15)',
'mdagger_5': 'Ensemble-DAgger-5',
'logger_1': 'BD-1',
'logger_5': 'BD-5',
'logger_25': 'BD-25',
'elogger_5': 'EBD-5',
'dagger_1_running_on_logger_5': 'DAgger on BD-5 Data',
'logger_5_running_on_dagger_1': 'BD-5 on DAgger Data'
}

color_mapping_dict = {
'expert': 'C0', # sky blue
'bc_1': 'C1',  #orange
'dagger_1': 'C3', # red
'logger_1': 'C4', #purple
'logger_5': 'C2', #green
'logger_25': 'C9', # sky blue
'mftpl_25': '#FF00FF', #magenta
'dagger_1_running_on_logger_5': 'C5', #brown
'logger_5_running_on_dagger_1': 'C6', #pink
'dagger_5': '#FF00FF', #magenta
'dagger_25': '#8B0000', #dark red
'mdagger_5': '#FFFF00', #yellow
'elogger_5': '#FFD700' #gold
}

# extra_color_dict_for_mix = {
# 'logger_5': '#BAF300', # yellow green for logger 5 mix     
# 'logger_25': '#F1BF99'  #skin color for logger 25 mix
# }

env_mapping_dict = {
'HalfCheetahBulletEnv-v0': 'Half Cheetah',    
'AntBulletEnv-v0': 'Ant',
'Walker2DBulletEnv-v0': 'Walker',
'HopperBulletEnv-v0': 'Hopper'
}


comp_dict = {
    'Hopper': {
        'bc_1': {'time': 442, 'memory': 1412},
        'dagger_1': {'time': 498, 'memory': 1412},
        'logger_1': {'time': 464, 'memory': 1412},
        'logger_5': {'time': 937, 'memory': 1414},
        'logger_25': {'time': 2508, 'memory': 1440},
        'mftpl_25':{'time': 2089, 'memory': 1438},
    },
    'Ant': {
        'bc_1': {'time': 552, 'memory': 1412},
        'dagger_1': {'time': 570, 'memory': 1412},
        'logger_1': {'time': 582, 'memory': 1412},
        'logger_5': {'time': 1136, 'memory': 1414},
        'logger_25': {'time': 2652, 'memory': 1440},
        'mftpl_25':{'time': 2338, 'memory': 1440},
    },
    'Half Cheetah': {
        'bc_1': {'time': 2775, 'memory': 1412},
        'dagger_1': {'time': 2741, 'memory': 1412},
        'logger_1': {'time': 2725, 'memory': 1412},
        'logger_5': {'time': 5136, 'memory': 1414},
        'logger_25': {'time': 14087, 'memory': 1440},
        'mftpl_25': {'time': 14211, 'memory': 1440},
    },
    'Walker': {
        'bc_1': {'time': 2576, 'memory': 1412},
        'dagger_1': {'time': 2611, 'memory': 1412},
        'logger_1': {'time': 2533, 'memory': 1412},
        'logger_5': {'time': 4487, 'memory': 1414},
        'logger_25': {'time': 12920, 'memory': 1440},
        'mftpl_25': {'time': 12344, 'memory': 1440},
    }
}



fig, axes = plt.subplots(2, 4, figsize=(20, 10)) 

# Plot settings
# task_order = ['Hopper', 'Ant', 'Cheetah', 'Walker']
colors = ['C1', 'C3', 'C4', 'C2', 'C9', '#FF00FF'] # Colors based on the name_ids

# Plotting the histograms
for i, name in enumerate(env_list):
    task = env_mapping_dict[name]
    # Extracting the data for each algorithm
    times = [comp_dict[task][algorithm]['time'] for algorithm in name_ids]
    memories = [comp_dict[task][algorithm]['memory'] for algorithm in name_ids]

    # # Plotting the time data
    # axes[0, i].bar( None , times, color=colors)
    # # axes[0, i].set_title(f'{task}')
    # axes[0, i].set_title(f'{task}', y=1.05)
    # # axes[0, i].set_xlabel('Algorithms')
    # axes[0, i].set_ylabel('Running Time (s)')

    x_values = range(len(times))
    axes[0, i].bar(x_values, times, color=colors)
    axes[0, i].set_title(f'{task}', y=1.05)  # Adjust the title position
    axes[0, i].set_ylabel('Running Time (s)')
    axes[0, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-ticks


    # Plotting the memory data
    axes[1, i].bar(label_ids, memories, color=colors)
    # axes[1, i].set_title(f'{task} Memory Cost')
    # axes[1, i].set_xlabel('Algorithms')
    axes[1, i].set_ylabel('Memory (MB)')

    # Rotating the x-ticks for better readability
    axes[0, i].tick_params(axis='x', rotation=45)
    axes[1, i].tick_params(axis='x', rotation=45)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

current_path = os.getcwd()
save_path = os.path.join(current_path, 'noisy_expert_rep_result')# noisy_expert_rep_result   noisy_expert_nonrealizable_results
os.makedirs(save_path, exist_ok=True)
plot_save_path = os.path.join(save_path, "cumputational_cost.png")
plt.savefig(plot_save_path)
plt.close()
