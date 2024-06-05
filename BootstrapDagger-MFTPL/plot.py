import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14

# env_list = ['HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HopperBulletEnv-v0']
env_list = ['HalfCheetahBulletEnv-v0', 'Walker2DBulletEnv-v0']
# env_list = ['AntBulletEnv-v0', 'HopperBulletEnv-v0']

name_ids = ['expert','bc_1', 'dagger_1', 'logger_1', 'logger_5', 'logger_25', 'mftpl_25_166'] #'mftpl_25_64' mftpl_25_166 for realizable, mftpl_25_128 for nonrealizable
# name_ids = ['bc_1', 'dagger_1', 'dagger_5', 'dagger_25','logger_1', 'logger_5','logger_25', 'mftpl_25_166'] # 'elogger_5',, 'elogger_25','mftpl_25_64', 'mftpl_25_166'

name_ids_for_ablation_study = ['expert', 'dagger_1',  'logger_5', 'logger_5_running_on_dagger_1', 'dagger_1_running_on_logger_5']#  , 'logger_25'

name_ids_for_ablation_study_2 = [ 'dagger_1', 'dagger_5', 'dagger_25', 'logger_1', 'logger_5',  'logger_25'] #, 'mdagger_5'  'expert',  , 'elogger_5'

# name_ids_for_ablation_study_mftpl = ['expert', 'logger_5','dagger_5', 'mftpl_5_16', 'dmftpl_5_sqrt_1', 'dmftpl_5_sqrt_05','dmftpl_5_sqrt_025'] 

# name_ids_for_ablation_study_mftpl = ['expert', 'logger_5','dagger_5', 'mftpl_5_16', 'dmftpl_5_8', 'dmftpl_5_16','dmftpl_5_32'] #, 'mftpl_5_2','mftpl_5_1'] 

# name_ids_for_ablation_study_mftpl = ['expert', 'dagger_1', 'mftpl_25_256','mftpl_25_128', 'mftpl_25_64', 'mftpl_25_32', 'mftpl_25_16', 'mftpl_25_8'] #, 'mftpl_5_2','mftpl_5_1'] 

# name_ids_for_ablation_study_mftpl = [ 'logger_25','dagger_25', 'mftpl_25_4', 'mftpl_25_8', 'mftpl_25_16'] #, 'mftpl_25_32', 'mftpl_25_64', 'mftpl_25_128'] #, 'mftpl_5_2','mftpl_5_1'] 

# name_ids_for_ablation_study_mftpl = ['expert', 'dagger_1', 'elogger_5','logger_5', 'mftpl_5_1000'] #,  'mftpl_5_16', 'mftpl_5_8', 'mftpl_5_4', 'mftpl_5_2'] #,'mftpl_5_1'] 

# name_ids_for_ablation_study_mftpl_25 = ['dagger_25', 'mftpl_25_64', 'mftpl_25_32', 'mftpl_25_16', 'mftpl_25_8']
# name_ids_for_ablation_study_mftpl_25 = ['expert', 'dagger_1', 'mftpl_25_256','mftpl_25_128', 'mftpl_25_64', 'mftpl_25_32', 'mftpl_25_16'] #, 'mftpl_25_8'] #, 'mftpl_5_2','mftpl_5_1'] 
# name_ids_for_ablation_study_mftpl_25 = ['expert', 'dagger_25', 'mftpl_25_128', 'mftpl_25_64', 'mftpl_25_32', 'mftpl_25_16', 'mftpl_25_8'] #, 'mftpl_5_2','mftpl_5_1'] 
name_ids_for_ablation_study_mftpl_25 = ['expert', 'dagger_25', 'mftpl_25_256', 'mftpl_25_128', 'mftpl_25_64', 'mftpl_25_32', 'mftpl_25_16'] #, 'mftpl_5_2','mftpl_5_1'] 
# name_ids_for_ablation_study_mftpl_25 = ['expert', 'dagger_25', 'mftpl_25_357', 'mftpl_25_166', 'mftpl_25_80', 'mftpl_25_40', 'mftpl_25_20']

name_ids_for_mix_vs_mean = ['logger_5', 'logger_25']

name_ids_no_expert = ['bc_1', 'dagger_1', 'logger_1', 'logger_5', 'logger_25','mftpl_25_166']

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
'dagger_25': 'MP-25(0)',#'DAgger-25',
'mdagger_5': 'Ensemble-DAgger-5',
'logger_1': 'BD-1',
'logger_5': 'BD-5',
'logger_25': 'BD-25',
'elogger_5': 'EBD-5',
'elogger_25': 'EBD-25',
'dagger_1_running_on_logger_5': 'SL on BD-5 Data',
'logger_5_running_on_dagger_1': 'Bagging on DAgger Data',
# 'mftpl_5_1': 'mftpl_2500',
# 'mftpl_5_2': 'mftpl_1250',
# 'mftpl_5_4': 'mftpl_625',
# 'mftpl_5_8': 'mftpl_312',
# 'mftpl_5_16': 'mftpl_156',
# 'mftpl_5_1000': 'mftpl_1',
# 'mftpl_5_1': 'mftpl_1000',
# 'mftpl_5_2': 'mftpl_500',
# 'mftpl_5_4': 'mftpl_250',
# 'mftpl_5_8': 'mftpl_125',
# 'mftpl_5_16': 'mftpl_62',
# 'mftpl_5_32': 'mftpl_31',
# 'mftpl_5_64': 'mftpl_15',
# 'mftpl_5_128': 'mftpl_7',
# 'mftpl_5_1000': 'mftpl_1',
# 'mftpl_25_1': 'mftpl_2500',
# 'mftpl_25_2': 'mftpl_1250',
# 'mftpl_25_4': 'mftpl_625',
# 'mftpl_25_8': 'mftpl_312',
# 'mftpl_25_16': 'mftpl_156',
# 'mftpl_25_1000': 'mftpl_2',
# 'mftpl_25_1': 'mftpl_2000',
# 'mftpl_25_2': 'mftpl_1000',
# 'mftpl_25_4': 'mftpl_500',
# 'mftpl_25_8': 'mftpl_250',
# 'mftpl_25_16': 'mftpl_125',
# 'mftpl_25_32': 'mftpl_62',
# 'mftpl_25_64': 'mftpl_31',
# # 'mftpl_25_64': 'mftpl_39',
'mftpl_25_8': 'MP-25(250)',
'mftpl_25_16': 'MP-25(125)',
'mftpl_25_32': 'MP-25(62)',
'mftpl_25_64': 'MP-25(31)', 
'mftpl_25_128': 'MP-25(15)',
'mftpl_25_256': 'MP-25(7)',
# 'mftpl_25_1000': 'mftpl_2',
# 'mftpl_25_8': 'MFTPL-P-25(125)',
# 'mftpl_25_16': 'MFTPL-P-25(62)',
# 'mftpl_25_32': 'MFTPL-P-25(31)',
# 'mftpl_25_64': 'MFTPL-P-25(15)', #25 ensemble 15 perturbation samples for realizable ant and hopper
# 'mftpl_25_8': 'MP-25(125)',
# 'mftpl_25_16': 'MP-25(62)',
# 'mftpl_25_32': 'MP-25(31)',
# 'mftpl_25_64': 'MP-25(15)', #25 ensemble 15 perturbation samples for realizable cheetah and walker
# 'mftpl_25_128': 'MP-25(7)', 
# 'mftpl_25_64': 'MFTPL-P-25(39)', #for cheetah and walker 25 ensemble 15 perturbation samples for realizable ant and hopper
'mftpl_25_20': 'MP-25(125)',
'mftpl_25_40': 'MP-25(62)',
'mftpl_25_80': 'MP-25(31)',
'mftpl_25_166': 'MP-25(15)',
'mftpl_25_357': 'MP-25(7)',
'dmftpl_5_8': 'dynamic mftpl 6-125',
'dmftpl_5_16': 'dynamic mftpl 3-62',
'dmftpl_5_32': 'dynamic mftpl 1-31',
'dmftpl_5_sqrt_1': 'dynamic mftpl sqrt 7-31',
'dmftpl_5_sqrt_05': 'dynamic mftpl sqrt 14-63',
'dmftpl_5_sqrt_025': 'dynamic mftpl sqrt 21-126'
}

color_mapping_dict = {
'expert': 'C0', # sky blue
'bc_1': 'C1',  #orange
'dagger_1': 'C3', # red
'logger_1': 'C4', #purple
'logger_5': 'C2', #green
'logger_25': 'C9', # sky blue
'dagger_1_running_on_logger_5': '#cdfa05', #yellow-green #'C5', #brown
'logger_5_running_on_dagger_1': 'C6', #pink
'dagger_5': '#FF00FF', #magenta
'dagger_25': '#8B0000', #dark red
'mdagger_5': '#FFFF00', #yellow
'elogger_5': '#FFD700', #gold
'elogger_25': '#8B0000', #dark red
'mftpl_5_1': 'C5', #brown
'mftpl_5_2': 'C6', #pink
'mftpl_5_4': '#FF00FF', #magenta
'mftpl_5_8': '#FFD700', #gold 
'mftpl_5_16': '#FFFF00', #yellow
'mftpl_5_32': 'C4', #purple
'mftpl_5_64': 'C5', #brown
'mftpl_5_128': 'C9', # sky blue
'mftpl_5_1000': '#8B0000', #dark red
'mftpl_25_1': 'C5', #brown
'mftpl_25_2': 'C6', #pink
'mftpl_25_4': '#FF00FF', #magenta
'mftpl_25_20': '#0781ad', #grey blue
'mftpl_25_40': '#504AF7', #blue purple
'mftpl_25_80': '#FFD700', #gold
'mftpl_25_166': '#FF00FF', #magenta
'mftpl_25_357': '#03fca1', #mint
'mftpl_25_16': '#0781ad', #grey blue
'mftpl_25_32': '#504AF7', #blue purple
'mftpl_25_64': '#FFD700', #gold
'mftpl_25_128': '#FF00FF', #magenta
'mftpl_25_256': '#03fca1', #mint
# 'mftpl_25_8': '#0781ad', #grey blue
# 'mftpl_25_16': '#504AF7', #blue purple
# 'mftpl_25_32': '#FFD700', #gold
# 'mftpl_25_64': '#FF00FF', #magenta
# 'mftpl_25_128': '#03fca1', #mint
'mftpl_25_1000': '#8B0000', #dark red
'dmftpl_5_8': 'C4', #purple
'dmftpl_5_16': 'C5', #brown
'dmftpl_5_32': 'C9', # sky blue
'dmftpl_5_sqrt_025': 'C4', #purple
'dmftpl_5_sqrt_05': 'C5', #brown
'dmftpl_5_sqrt_1': 'C9' # sky blue
}
# '#797BB7', # purple blue
# '#F52589', # pink red
extra_color_dict_for_mix = {
'logger_5': '#BAF300', # yellow green for logger 5 mix     
'logger_25': '#F1BF99'  #skin color for logger 25 mix
}

env_mapping_dict = {
'HalfCheetahBulletEnv-v0': 'Half Cheetah',    
'AntBulletEnv-v0': 'Ant',
'Walker2DBulletEnv-v0': 'Walker',
'HopperBulletEnv-v0': 'Hopper'
}

def bootstrap_confidence_bounds(data, n_bootstrap=1000, alpha=0.2):
    """
    data: 1D array (len 10)
    alpha: CB significance level
    """
    bootstrap_means = [np.mean(np.random.choice(data, len(data))) for _ in range(n_bootstrap)]
    low = np.percentile(bootstrap_means, 100 * (alpha / 2))
    up = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    lower_bound = 2*np.mean(data)-up
    upper_bound = 2*np.mean(data)-low

    return lower_bound, upper_bound


def plot_reward_results(result_folder, env_name, ablation):

    # env_name =  'HopperBulletEnv-v0' 
    # # 'HopperBulletEnv-v0'   'AntBulletEnv-v0'   'HalfCheetahBulletEnv-v0'    'Walker2DBulletEnv-v0'  
    if not ablation:
        id_list = name_ids
    elif ablation==1:
        id_list = name_ids_for_ablation_study
    elif ablation==2:
        id_list = name_ids_for_ablation_study_2
    elif ablation==3:
        id_list = name_ids_for_ablation_study_mftpl
    elif ablation==4:
        id_list = name_ids_for_ablation_study_mftpl_25
    current_path = os.getcwd()
    save_path = os.path.join(current_path, result_folder, env_name)# noisy_expert_rep_result   noisy_expert_nonrealizable_results

    plt.figure()
    
    for save_name_id in id_list:
        full_save_path = os.path.join(save_path, f"{save_name_id}.pkl")
        color = color_mapping_dict[save_name_id]
        with open(full_save_path, 'rb') as f:
            results_dict = pickle.load(f)
        if save_name_id != 'expert':
            mean_result = np.mean(results_dict['concatenated_result'], axis=1)
            lower_bounds = []
            upper_bounds = []
            for row in results_dict['concatenated_result']:
                lower, upper = bootstrap_confidence_bounds(row)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            plt.plot(results_dict['datasize_list'], mean_result, label=f"{name_mapping_dict[save_name_id]}", color=color)
            plt.fill_between(results_dict['datasize_list'], lower_bounds, upper_bounds, color=color, alpha=0.2)
        else:
            mean_result = np.mean(results_dict['concatenated_result'], axis=0)
            lower, upper = bootstrap_confidence_bounds(results_dict['concatenated_result'])
            if 'nonrealizable' in save_path or 'linear' in save_path:
                plot_size = 50 if env_name in ['HalfCheetahBulletEnv-v0','Walker2DBulletEnv-v0'] else 40
            else:
                plot_size = 50 if env_name in ['HalfCheetahBulletEnv-v0','Walker2DBulletEnv-v0'] else 20
            mean_result = np.full((plot_size,), mean_result)
            lower_bounds = np.full((plot_size,), lower)
            upper_bounds = np.full((plot_size,), upper)
            with open(os.path.join(save_path, f"{'dagger_1'}.pkl"), 'rb') as f:
                dagger_dict = pickle.load(f)
            sample_dict = dagger_dict['datasize_list']
            plt.plot(sample_dict, mean_result, label=f"{name_mapping_dict[save_name_id]}", color=color)
            plt.fill_between(sample_dict, lower_bounds, upper_bounds, color=color, alpha=0.2)
    if 'nonrealizable' in save_path:
        plt.title(f'{env_mapping_dict[env_name]} (non-realizable expert)')
    elif 'linear' in save_path:
        # plt.title(f'{env_mapping_dict[env_name]} (linear model)')
        plt.title(r'{} (linear model uniform $d_0$)'.format(env_mapping_dict[env_name]))
    else:
        plt.title(env_mapping_dict[env_name])
    plt.xlabel('Number of Expert Annotations')
    plt.ylabel('Test Reward Value')
    plt.legend() #loc='upper left'
    # plt.legend(loc='upper left')
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 0.91)) #for nonrealizable walker
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 0.93)) #for nonrealizable ant 
    plt.show()
    
    overall_plot_path = os.path.join(save_path, 'overall_plots')
    os.makedirs(overall_plot_path, exist_ok=True)
    print(overall_plot_path)

    if not ablation:
        if 'nonrealizable' in save_path:
            plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_nonrealizable_plot.png")
        elif 'linear' in save_path:
            plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_linear_plot.png")
        else:
            plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_main_plot.png")
    elif ablation ==1:
        if 'nonrealizable' in save_path:
            sys.exit()
        else:
            plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_ablation_study.png")
    elif ablation ==2:
        if 'nonrealizable' in save_path:
            sys.exit()
        else:
            plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_ensemble_dagger.png")
    elif ablation ==3:
        if 'nonrealizable' in save_path:
            sys.exit()
        else:
            plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_mftpl_5_ablation.png")
    elif ablation ==4:
        if 'nonrealizable' in save_path:
            plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_mftpl_25_nonrealizable.png")
        else:
            plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_mftpl_25_ablation.png")
    
    plt.savefig(plot_save_path)
    plt.close()

def plot_mix_vs_mean(result_folder,env_name):
    id_list = name_ids_for_mix_vs_mean
    current_path = os.getcwd()
    save_path = os.path.join(current_path, result_folder, env_name)# noisy_expert_rep_result   noisy_expert_nonrealizable_results

    plt.figure()
    
    for save_name_id in id_list:
        full_save_path = os.path.join(save_path, f"{save_name_id}.pkl")
        color = color_mapping_dict[save_name_id]
        if save_name_id != 'expert':
            color_mix = extra_color_dict_for_mix[save_name_id]
        with open(full_save_path, 'rb') as f:
            results_dict = pickle.load(f)
        if save_name_id != 'expert':
            mean_result = np.mean(results_dict['concatenated_result'], axis=1)
            lower_bounds = []
            upper_bounds = []
            for row in results_dict['concatenated_result']:
                lower, upper = bootstrap_confidence_bounds(row)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            plt.plot(results_dict['datasize_list'], mean_result, label=f"{name_mapping_dict[save_name_id]}", color=color)
            plt.fill_between(results_dict['datasize_list'], lower_bounds, upper_bounds, color=color, alpha=0.2)

            random_mean_result = np.mean(results_dict['concatenated_random_result'], axis=1)
            random_lower_bounds = []
            random_upper_bounds = []
            for row in results_dict['concatenated_random_result']:
                lower, upper = bootstrap_confidence_bounds(row)
                random_lower_bounds.append(lower)
                random_upper_bounds.append(upper)
            plt.plot(results_dict['datasize_list'], random_mean_result, label=f"{name_mapping_dict[save_name_id]} random", color=color_mix)
            # plt.fill_between(results_dict['datasize_list'], random_lower_bounds, random_upper_bounds, color=color_mix, alpha=0.2)
        else:
            mean_result = np.mean(results_dict['concatenated_result'], axis=0)
            lower, upper = bootstrap_confidence_bounds(results_dict['concatenated_result'])
            if 'nonrealizable' in save_path or 'linear' in save_path:
                plot_size = 50 if env_name in ['HalfCheetahBulletEnv-v0','Walker2DBulletEnv-v0'] else 40
            else:
                plot_size = 50 if env_name in ['HalfCheetahBulletEnv-v0','Walker2DBulletEnv-v0'] else 20
            mean_result = np.full((plot_size,), mean_result)
            lower_bounds = np.full((plot_size,), lower)
            upper_bounds = np.full((plot_size,), upper)
            with open(os.path.join(save_path, f"{'dagger_1'}.pkl"), 'rb') as f:
                dagger_dict = pickle.load(f)
            sample_dict = dagger_dict['datasize_list']
            plt.plot(sample_dict, mean_result, label=f"{name_mapping_dict[save_name_id]}", color=color)
            plt.fill_between(sample_dict, lower_bounds, upper_bounds, color=color, alpha=0.2)
    if 'nonrealizable' in save_path:
        plt.title(f'{env_mapping_dict[env_name]} (non-realizable expert)')
    else:
        plt.title(env_mapping_dict[env_name])
    plt.xlabel('Number of Expert Annotations')
    plt.ylabel('Test Reward Value')
    plt.legend()
    plt.show()
    
    overall_plot_path = os.path.join(save_path, 'overall_plots')
    os.makedirs(overall_plot_path, exist_ok=True)
    print(overall_plot_path)


    plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_mix_vs_mean.png")
    plt.savefig(plot_save_path)
    plt.close()


def plot_imitation_loss(result_folder,env_name):
    # env_name = 'HalfCheetahBulletEnv-v0'
    # # 'HopperBulletEnv-v0'   'AntBulletEnv-v0'   'HalfCheetahBulletEnv-v0'    'Walker2DBulletEnv-v0'  

    current_path = os.getcwd()
    save_path = os.path.join(current_path, result_folder, env_name)# noisy_expert_nonrealizable_results noisy_expert_rep_result

    plt.figure()
    
    for save_name_id in name_ids_no_expert:
        full_save_path = os.path.join(save_path, f"{save_name_id}.pkl")
        color = color_mapping_dict[save_name_id]
        with open(full_save_path, 'rb') as f:
            results_dict = pickle.load(f)

        mean_result = np.mean(results_dict['concatenated_loss'], axis=1)
        lower_bounds = []
        upper_bounds = []
        for row in results_dict['concatenated_loss']:
            lower, upper = bootstrap_confidence_bounds(row)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        plt.plot(results_dict['datasize_list'], mean_result, label=f"{name_mapping_dict[save_name_id]}", color=color)
        plt.fill_between(results_dict['datasize_list'], lower_bounds, upper_bounds, color=color, alpha=0.2)
    if 'nonrealizable' not in save_path:    
        plt.title(env_mapping_dict[env_name])
    else:
        plt.title(f'{env_mapping_dict[env_name]} (non-realizable expert)')
    plt.xlabel('Number of Expert Annotations')
    plt.ylabel('Imitation Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    
    plt.show()
    
    overall_plot_path = os.path.join(save_path, 'overall_plots')
    os.makedirs(overall_plot_path, exist_ok=True)
    print(overall_plot_path)

    if 'nonrealizable' in save_path:
        plot_name = env_mapping_dict[env_name]+"_nonrealizable_imitation_loss.png"
    else:
        plot_name = env_mapping_dict[env_name]+"_imitation_loss.png"
    plot_save_path = os.path.join(overall_plot_path, plot_name)
    plt.savefig(plot_save_path)
    plt.close()



def assign_colors_to_data(data, n_groups=50):
    n_samples = data.shape[0]
    colors = np.linspace(0, 1, n_samples)
    
    # Split the colors array into `n_groups` of equal size
    groups = np.array_split(colors, n_groups)
    
    # Calculate the mean color for each group
    group_colors = [np.mean(g) for g in groups]
    
    # Expand these mean colors to assign to each individual sample
    expanded_colors = np.concatenate([[c] * len(g) for c, g in zip(group_colors, groups)])
    
    # Create a colormap that goes from blue to red
    colormap = plt.cm.get_cmap("RdYlBu_r", n_groups)
    
    return colormap(expanded_colors)



def tsne_single_visualization(result_folder,env_name,alg_name):
    # env_name = 'AntBulletEnv-v0'
    # alg_name = 'logger_1'
    # 'HalfCheetahBulletEnv-v0'  # 'AntBulletEnv-v0'  # 'Walker2DBulletEnv-v0'  'HopperBulletEnv-v0'

    current_path = os.getcwd()
    save_path = os.path.join(current_path, result_folder, env_name)
    tsne_save_path = os.path.join(current_path, result_folder, env_name, "tsne_full_dataset_new.npy")
    if 'nonrealizable' in save_path:
        n_groups = 50 if env_name in ['HalfCheetahBulletEnv-v0','Walker2DBulletEnv-v0'] else 40
    else:
        n_groups = 50 if env_name in ['HalfCheetahBulletEnv-v0','Walker2DBulletEnv-v0'] else 20

    # Check if the transformed dataset already exists
    if os.path.exists(tsne_save_path):
        print("Loading existing T-SNE transformed data...")
        full_dataset_transformed = np.load(tsne_save_path)
    else:
        full_dataset = None
        for i, save_name_id in enumerate(name_ids_no_expert):
            full_save_path = os.path.join(save_path, f"{save_name_id}.pkl")
            
            with open(full_save_path, 'rb') as f:
                results_dict = pickle.load(f)

            obs_data = results_dict['concatenated_obs']
            reshaped_data = obs_data.reshape(-1, obs_data.shape[-1])  # Reshape each dataset
            
            if full_dataset is None:  # If the full_dataset is empty, initialize it
                full_dataset = reshaped_data
            else:  # Otherwise, concatenate
                full_dataset = np.concatenate([full_dataset, reshaped_data], axis=0)
        print('all data',full_dataset.shape)
    
        print("Fitting T-SNE to full dataset")
        tsne = TSNE(n_components=2, random_state=42)
        full_dataset_transformed = tsne.fit_transform(full_dataset)
        print('saving transformed dataset')
        np.save(tsne_save_path, full_dataset_transformed)

    start_idx = 0


    for i, save_name_id in enumerate(name_ids_no_expert):
        full_save_path = os.path.join(save_path, f"{save_name_id}.pkl")

        with open(full_save_path, 'rb') as f:
            results_dict = pickle.load(f)

        # for key, value in results_dict.items():
        #     print(key)

        obs_data = results_dict['concatenated_obs']
        reshaped_data = obs_data.reshape(-1, obs_data.shape[-1])

        end_idx = start_idx + reshaped_data.shape[0]
        
        if alg_name == save_name_id:
            all_transformed_data = full_dataset_transformed[start_idx:end_idx, :]
        
        start_idx = end_idx  # Update the starting index for the next iteration

    colors = assign_colors_to_data(all_transformed_data,n_groups)

    plt.figure(figsize=(10, 10))

    plt.scatter(all_transformed_data[:, 0], all_transformed_data[:, 1], alpha=0.5, c=colors, s=2, label=name_mapping_dict[alg_name])
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Distribution of Observations for {name_mapping_dict[alg_name]} on {env_mapping_dict[env_name]}")

    # # Create an axis for the horizontal colorbar
    colorbar_map = plt.cm.get_cmap("RdYlBu_r", n_groups)
    cax = plt.axes([0.68, 0.15, 0.20, 0.03])  # Here, [left, bottom, width, height] are in figure coordinates.
    norm = Normalize(vmin=0, vmax=n_groups)
    sm = ScalarMappable(cmap=colorbar_map, norm=norm)
    sm.set_array([])

    # Add the colorbar to the created axis
    # cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    # cbar.set_label('Data Arrival Order', labelpad=1.5)
    # cbar.set_ticks([0, n_groups])
    # cbar.ax.tick_params(axis='x', direction='out', pad=-13)
    # cbar.ax.set_xticklabels(['Earlier', 'Later'], ha="center", position=(0, 1.5))
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Data Arrival Rounds',fontsize=16,labelpad=5.5)  # Adjust labelpad for more distance
    cbar.set_ticks([0, n_groups])
    # cbar.ax.tick_params(axis='x')  # Adjust pad value for more distance

    # Set tick labels below the bar and center-align them
    # cbar.ax.set_xticklabels(['Earlier', 'Later'], ha="center", position=(0, -1))

    # Position the label on top
    cbar.ax.xaxis.set_label_position('top') 

    plt.show()


    overall_plot_path = os.path.join(save_path, 'visualization')
    os.makedirs(overall_plot_path, exist_ok=True)
    
    plot_save_path = os.path.join(overall_plot_path, f"{env_mapping_dict[env_name]}_{alg_name}_tsne.png")
    plt.savefig(plot_save_path)
    plt.close()


def main(result_folder):

    plot_reward_results(result_folder, 'AntBulletEnv-v0', ablation=4)   #2,4
    # plot_imitation_loss(result_folder,'Walker2DBulletEnv-v0')
    # for env_name in env_list:#env_list  'HopperBulletEnv-v0', 'AntBulletEnv-v0' 'Walker2DBulletEnv-v0' 'HalfCheetahBulletEnv-v0' 
        # plot_reward_results(result_folder, env_name, ablation=3)
        # plot_imitation_loss(result_folder,env_name)
        # plot_mix_vs_mean(result_folder,env_name)
        # if not 'nonrealizable' in result_folder:
        #     plot_reward_results(result_folder, env_name, ablation=True)
            # plot_imitation_loss(result_folder,env_name)
            # for alg_name in name_ids_no_expert:
            #     print(result_folder,env_name,alg_name)
            #     tsne_single_visualization(result_folder,env_name,alg_name)
        
# def main(result_folder):

#     for env_name in env_list:#env_list  'HopperBulletEnv-v0', 'AntBulletEnv-v0' 'Walker2DBulletEnv-v0' 'HalfCheetahBulletEnv-v0' 
#         # plot_reward_results(result_folder, env_name, ablation=False)
#         # plot_mix_vs_mean(result_folder,env_name)
#         # plot_imitation_loss(result_folder,env_name)
#         if not 'nonrealizable' in result_folder:
#         #     plot_reward_results(result_folder, env_name, ablation=True)
#             for alg_name in name_ids_no_expert:
#                 print(result_folder,env_name,alg_name)
#                 tsne_single_visualization(result_folder,env_name,alg_name)

if __name__ == "__main__":
    # result_folder = 'noisy_expert_rep_result'
    # noisy_expert_rep_result   noisy_expert_nonrealizable_results noisy_expert_linear_rep_result
    main(result_folder = 'noisy_expert_linear_rep_result')


