# from soil_function import soil_function
from soil_function_mftpl import soil_function
import os
import signal
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

def handler(signum, frame):
    raise Exception("Soil function takes too long, probably stucked at evaluation")

def loop_run(saved_folder,algorithm,ensemble_size,env_name,device="cuda:0",additional_sample_rate=8,noisy_expert=True,random_selection = True,non_realizable=False,linear = False):
    results_dict = {'datasize_list': None, 'policy_list': None,
                    'concatenated_result': None, 'concatenated_result_std': None, 'concatenated_loss': None, 'concatenated_loss_std': None, 
                    'concatenated_random_result': None, 'concatenated_random_result_std': None, 'concatenated_random_loss': None, 'concatenated_random_loss_std': None,
                    'concatenated_obs': None, 'concatenated_acs': None
                    }
    

    rl_baseline_zoo_dir = 'xxxx/rl-baselines-zoo'
    save_name_id = algorithm + '_' + str(ensemble_size)
    
    if algorithm == 'mftpl':
        save_name_id = algorithm + '_' + str(ensemble_size) + '_' + str(additional_sample_rate)

    if algorithm == 'dmftpl':
        save_name_id = algorithm + '_' + str(ensemble_size) + '_sqrt_' + str(additional_sample_rate).replace('.', '')
        # save_name_id = algorithm + '_' + str(ensemble_size) + '_' + str(additional_sample_rate)


    # if algorithm == 'dagger':
    #     random_selection = False
    # else:
    #     random_selection = True

    # Get path
    current_path = os.getcwd()
    save_path = os.path.join(current_path, saved_folder, env_name) #noisy_expert_nonrealizable_results  #noisy_expert_rep_result
    os.makedirs(save_path, exist_ok=True)
    save_name = save_name_id + '.pkl' 

    full_save_path = os.path.join(save_path, save_name)

    signal.signal(signal.SIGALRM, handler)

    try:
        with open(full_save_path, 'rb') as f:
            results_dict = pickle.load(f)
            existing_runs = results_dict['concatenated_result'].shape[1]
            print('existing_runs',existing_runs)
            # print(results_dict['policy_list'])
    except FileNotFoundError:
        existing_runs = 0
    
    if existing_runs == 10:
        print('10 trajectory already exist')
        return

    # Run soil_function until 10 runs
    while existing_runs < 10:

        # Start the 6-hour timer
        signal.alarm(6 * 60 * 60)

        try:
            print('existing_runs',existing_runs)
            print('!!!non realizable',non_realizable,' linear model', linear)
            # datasize_list, policies, result, random_result, obs, acs = soil_function(algorithm,ensemble_size,env_name=env_name, device = device,seed = existing_runs+1,random_selection=random_selection,noisy_expert = noisy_expert,non_realizable=non_realizable,rl_baseline_zoo_dir=rl_baseline_zoo_dir)
            datasize_list, policies, result, random_result, obs, acs = soil_function(algorithm,ensemble_size,env_name=env_name, device = device,seed = existing_runs+1,random_selection=random_selection,noisy_expert = noisy_expert,additional_sample_rate=additional_sample_rate,non_realizable=non_realizable,linear=linear,additional_data_folder = saved_folder,rl_baseline_zoo_dir=rl_baseline_zoo_dir)
            
            
        #     soil_function(
        # algorithm=algorithm,ensemble_size=ensemble_size,env_name=  env_name ,device = device, noisy_expert=True,additional_sample_rate=additional_sample_rate, non_realizable = non_realizable) #,noisy_expert=False, random_selection=True,non_realizable = False

            
            # Turn off the timer
            signal.alarm(0)
            
            
            # result [0] rollout result, [1] rollout std, [2] imitation loss, [3] imitation loss std

            if existing_runs == 0:
                results_dict['datasize_list'] = datasize_list
                results_dict['policy_list'] = [policies]
                results_dict['concatenated_result'] = np.expand_dims(result[0], axis=1)
                results_dict['concatenated_result_std'] = np.expand_dims(result[1], axis=1)
                results_dict['concatenated_loss'] = np.expand_dims(result[2], axis=1)
                results_dict['concatenated_loss_std'] = np.expand_dims(result[3], axis=1)
                if ensemble_size > 1:
                    results_dict['concatenated_random_result'] = np.expand_dims(random_result[0], axis=1)
                    results_dict['concatenated_random_result_std'] = np.expand_dims(random_result[1], axis=1)
                    results_dict['concatenated_random_loss'] = np.expand_dims(random_result[2], axis=1)
                    results_dict['concatenated_random_loss_std'] = np.expand_dims(random_result[3], axis=1)
                results_dict['concatenated_obs'] = np.expand_dims(obs, axis=1)
                results_dict['concatenated_acs'] = np.expand_dims(acs, axis=1)
    
            else:
                results_dict['policy_list'].append(policies)
                # Concatenate results along a new dimension
                results_dict['concatenated_result'] = np.concatenate([results_dict['concatenated_result'], np.expand_dims(result[0], axis=1)], axis=1)
                results_dict['concatenated_result_std'] = np.concatenate([results_dict['concatenated_result_std'], np.expand_dims(result[1], axis=1)], axis=1)
                results_dict['concatenated_loss'] = np.concatenate([results_dict['concatenated_loss'], np.expand_dims(result[2], axis=1)], axis=1)
                results_dict['concatenated_loss_std'] = np.concatenate([results_dict['concatenated_loss_std'], np.expand_dims(result[3], axis=1)], axis=1)
                if ensemble_size > 1:
                    results_dict['concatenated_random_result'] = np.concatenate([results_dict['concatenated_random_result'], np.expand_dims(random_result[0], axis=1)], axis=1)
                    results_dict['concatenated_random_result_std'] = np.concatenate([results_dict['concatenated_random_result_std'], np.expand_dims(random_result[1], axis=1)], axis=1)
                    results_dict['concatenated_random_loss'] = np.concatenate([results_dict['concatenated_random_loss'], np.expand_dims(random_result[2], axis=1)], axis=1)
                    results_dict['concatenated_random_loss_std'] = np.concatenate([results_dict['concatenated_random_loss_std'], np.expand_dims(random_result[3], axis=1)], axis=1)
                results_dict['concatenated_obs'] = np.concatenate([results_dict['concatenated_obs'],np.expand_dims(obs, axis=1)], axis=1)
                results_dict['concatenated_acs'] = np.concatenate([results_dict['concatenated_acs'],np.expand_dims(acs, axis=1)], axis=1)
            existing_runs += 1

            # Save results as a dictionary
            with open(full_save_path, 'wb') as f:
                pickle.dump(results_dict, f)
            
            print(f"Results saved at {full_save_path}")

        except Exception as e:
            print(f"An timeout exception occurred: {e}")
            sys.stdout.flush()


    if existing_runs==10:
        # for key, value in results_dict.items():
        #     if key != 'policy_list':
        #         print(f"Shape of {key}: {value.shape}")
        #         print(f"Values of {key}: {value}")

        plot_save_path = os.path.join(save_path, 'naive_plots')
        os.makedirs(plot_save_path, exist_ok=True)

        mean_result = np.mean(results_dict['concatenated_result'], axis=1)
        std_result = np.std(results_dict['concatenated_result'], axis=1)

        # mean_rollout_std = np.mean(results_dict['concatenated_result_std'], axis=1)
        mean_loss = np.mean(results_dict['concatenated_loss'], axis=1)
        std_loss = np.std(results_dict['concatenated_loss'], axis=1)
        
        # Plot mean and std of concatenated_result
        plt.figure()
        plt.errorbar(results_dict['datasize_list'], mean_result, yerr=std_result, fmt='o-', label='Mean reward & std')
        plt.title('Mean and Standard Deviation of Results')
        plt.xlabel('Data Size')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        plot_save_path1 = os.path.join(plot_save_path, f"plot1_{save_name_id}.png")
        plt.savefig(plot_save_path1)
        plt.close()

        # Plot mean of concatenated_result and mean of concatenated_result_std
        plt.figure()
        plt.errorbar(results_dict['datasize_list'], mean_loss,  yerr= std_loss ,fmt = 'x-', label='Mean loss & std')
        plt.title('Mean of Results and Mean of Rollout Std')
        plt.xlabel('Data Size')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plot_save_path2 = os.path.join(plot_save_path, f"plot2_{save_name_id}.png")
        plt.savefig(plot_save_path2)
        plt.close()


if __name__ == "__main__":
    start_time = time.time()

    saved_folder = 'noisy_expert_nonrealizable_results'  #noisy_expert_nonrealizable_results   noisy_expert_linear_rep_result noisy_expert_rep_result noisy_expert_linear_result   
    algorithm = 'dagger'
    # 'bc', 'dagger', 'logger', 'elogger', 'mftpl'
    ensemble_size = 25
    env_name =  'Walker2DBulletEnv-v0'
    #  'HopperBulletEnv-v0'   'AntBulletEnv-v0'   'HalfCheetahBulletEnv-v0'    'Walker2DBulletEnv-v0'  
    device = "cuda:2"
    additional_sample_rate = 357
    noisy_expert=True
    random_selection = True
    non_realizable = True
    linear = False


    loop_run(saved_folder=saved_folder,algorithm=algorithm,ensemble_size=ensemble_size,env_name=env_name,device=device,additional_sample_rate=additional_sample_rate,noisy_expert=noisy_expert,random_selection =random_selection,non_realizable=non_realizable,linear=linear)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    print(f"Algorithm: {algorithm}")
    print(f"Ensemble size: {ensemble_size}")
    print(f"Environment name: {env_name}")