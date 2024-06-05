import gym, os
import numpy as np
import argparse
import random
import pandas as pd
import copy

import sys
import torch
from gym import wrappers
import random
import torch.nn.functional as F
import torch.nn as nn
import torch as th

from dril.a2c_ppo_acktr.envs import make_vec_envs
from dril.a2c_ppo_acktr.model import Policy
from dril.a2c_ppo_acktr.arguments import get_args
import dril.a2c_ppo_acktr.ensemble_models as ensemble_models
from eval_ensemble import eval_ensemble_class
import os

from mftpl import mftpl
from copy import deepcopy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env_to_hiddensize_dict = {
    'HopperBulletEnv-v0': 4, 
    'AntBulletEnv-v0': 8,
    'HalfCheetahBulletEnv-v0': 12,  
    'Walker2DBulletEnv-v0': 16   
    }

def custom_format(num):
    if num >= 1:
        return f"{num:.6f}"
    else:
        return f"{num:.3e}"

def run_single_trail(dataset_name,obs_dataset,acs_dataset,algorithm,ensemble_size,env_name,device,seed =1,non_realizable=False,rl_baseline_zoo_dir='/home/user/Downloads/home/yichen/dril/rl-baselines-zoo'):
    args = get_args()

    args.behavior_cloning = True if algorithm=='bc' else False
    args.ensemble_shuffle_type = 'bootstrap' if algorithm=='logger' else 'norm_shuffle'

    args.ensemble_size = ensemble_size
    args.device = device
    if non_realizable:
        args.rounds = 40 if env_name in ['AntBulletEnv-v0','HopperBulletEnv-v0'] else 50
    else:
        args.rounds = 20 if env_name in ['AntBulletEnv-v0','HopperBulletEnv-v0'] else 50
    args.non_realizable = non_realizable
    args.seed = seed

    print('Env Name: ',env_name, 'rounds: ', args.rounds)

    if args.non_realizable:
        args.hidden_size = env_to_hiddensize_dict[env_name]

    if args.behavior_cloning:
        print("bc",args.ensemble_size)
        save_name_id = "bc_"+str(args.ensemble_size)
    elif args.ensemble_shuffle_type=='norm_shuffle':
        print("dagger",args.ensemble_size)
        save_name_id = "dagger_"+str(args.ensemble_size)
    else:
        print("logger",args.ensemble_size)
        save_name_id = "logger_"+str(args.ensemble_size)


    args.env_name = env_name
    args.rl_baseline_zoo_dir = rl_baseline_zoo_dir

    args.recurrent_policy = False
    args.load_expert = True

    os.system(f'mkdir -p {args.demo_data_dir}')
    os.system(f'mkdir -p {args.demo_data_dir}/tmp/gym')
    sys.path.insert(1,os.path.join(args.rl_baseline_zoo_dir, 'utils'))
    from a2c_ppo_acktr.utils import get_saved_hyperparams

    #device = torch.device("cpu")
    device = torch.device(args.device if args.cuda else "cpu")
    print(f'device: {device}')
    # seed = args.seed
    print(f'seed: {args.seed}')

    if args.env_name in ['highway-v0']:
        raise NotImplementedError
        import highway_env
        from rl_agents.agents.common.factory import agent_factory

        env = make_vec_envs(args.env_name, seed, 1, 0.99, f'{args.emo_data_dir}/tmp/gym', device,\
                        True, stats_path=stats_path, hyperparams=hyperparams, time=time,
                        atari_max_steps=args.atari_max_steps)
        # envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
        #                      args.log_dir, device, False, use_obs_norm=args.use_obs_norm,
        #                      max_steps=args.atari_max_steps)
        agent_config = {
            "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
            "budget": args.data_per_round,
            "gamma": 0.7,
        }
        th_model = agent_factory(gym.make(args.env_name), agent_config)
        time = False
    elif args.env_name in ['duckietown']:
        raise NotImplementedError
        from a2c_ppo_acktr.duckietown.env import launch_env
        from a2c_ppo_acktr.duckietown.wrappers import NormalizeWrapper, ImgWrapper,\
            DtRewardWrapper, ActionWrapper, ResizeWrapper
        from a2c_ppo_acktr.duckietown.teacher import PurePursuitExpert
        env = launch_env()
        env = ResizeWrapper(env)
        env = NormalizeWrapper(env)
        env = ImgWrapper(env)
        env = ActionWrapper(env)
        env = DtRewardWrapper(env)

        # Create an imperfect demonstrator
        expert = PurePursuitExpert(env=env)
        time = False
    else:
        print('[Setting environemnt hyperparams variables]')

        if args.env_name in ['AntBulletEnv-v0']:
            args.expert_algo = 'trpo'
        else:
            args.expert_algo = 'ppo2'
            
        stats_path = os.path.join(args.rl_baseline_zoo_dir, 'trained_agents', f'{args.expert_algo}',\
                            f'{args.env_name}')
        hyperparams, stats_path = get_saved_hyperparams(stats_path, test_mode=True,\
                                            norm_reward=args.norm_reward_stable_baseline)

        ## Load saved policy

        # subset of the environments have time wrapper
        time_wrapper_envs = ['HalfCheetahBulletEnv-v0', 'Walker2DBulletEnv-v0', 'AntBulletEnv-v0']
        if args.env_name in time_wrapper_envs:
            time=True
            print('use time as feature')
        else:
            time = False

        env = make_vec_envs(args.env_name, args.seed, 1, 0.99, f'{args.demo_data_dir}/tmp/gym', device,\
                        True, stats_path=stats_path, hyperparams=hyperparams, time=time)#time
        env.reset()
        th_model = Policy(
            env.observation_space.shape,
            env.action_space,
            load_expert=True,
            env_name=args.env_name,
            rl_baseline_zoo_dir=args.rl_baseline_zoo_dir,
            expert_algo=args.expert_algo,
            # [Bug]: normalize=False,
            normalize=True if hasattr(gym.envs, 'atari') else False,
            base_kwargs={'recurrent': args.recurrent_policy}).to(device)
        th_model.dist = th_model.dist.to(device)
        expert_param = copy.deepcopy(th_model.state_dict())

        env.close()
    


    saved_param = None
        
    # #define ensemble policy    
    ensemble_size = args.ensemble_size
    try:
        num_actions = env.action_space.n
    except:
        num_actions = env.action_space.shape[0]

    print('hidden size',args.hidden_size)
    ensemble_args = (env.observation_space.shape[0], num_actions, args.hidden_size, ensemble_size)
    # ensemble_args = (env.observation_space.shape[0], num_actions,32, ensemble_size)


    
    if len(env.observation_space.shape) == 3:
        if args.env_name in ['duckietown']:
            policy_def = ensemble_models.PolicyEnsembleDuckieTownCNN
        else:
            policy_def = ensemble_models.PolicyEnsembleCNN
    else:
        if args.non_realizable:
            policy_def = ensemble_models.PolicyEnsembleMLP_nonrealizable
        else:
            policy_def = ensemble_models.PolicyEnsembleMLP_simple
        
    ensemble_policy = policy_def(*ensemble_args).to(device)

    saved_param = copy.deepcopy(ensemble_policy.state_dict())

    # set evaluation method
    eval = eval_ensemble_class(ensemble_size, None, args.env_name, args.seed,
                            args.num_processes, None, device, num_episodes=args.num_processes,
                            stats_path=stats_path, hyperparams=hyperparams, time=time)
    
    policy_list = []
    result_list = []
    std_list = []
    loss_list = []
    loss_std_list = []

    random_result_list = []
    random_std_list = []
    random_loss_list = []
    random_loss_std_list = []
    for rounds in range(args.rounds):
        set_size = int((rounds+1)*args.data_per_round)
        rtn_obs_ = obs_dataset[:set_size]
        rtn_acs_ = acs_dataset[:set_size]

        # ensemble_param, result, std, loss = mftpl(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_),len(rtn_obs), stats_path=stats_path, hyperparams=hyperparams, time=time )
        ensemble_param = mftpl(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_))
        saved_param = deepcopy(ensemble_param)
        ensemble_policy.load_state_dict(saved_param)
        result,std,loss, loss_std = eval.test(ensemble_policy,expert = th_model,random_selection=False)

        if args.ensemble_size > 1:
            random_result,random_std,random_loss,random_loss_std = eval.test(ensemble_policy,expert = th_model,random_selection=True)
            print(
                save_name_id, 
                f'{set_size} {dataset_name} samples avg: {custom_format(result)}, std: {custom_format(std)},'
                f'loss: {custom_format(loss)},loss_std: {custom_format(loss_std)}, '
                f'random: {custom_format(random_result)}, std: {custom_format(random_std)}, '
                f'loss:{custom_format(random_loss)}, std: {custom_format(random_loss_std)}'
            )
            # print(save_name_id, f'{len(rtn_obs)} samples avg: {custom_format(result)}, std: {custom_format(std)},loss: {custom_format(loss)},loss_std: {custom_format(loss_std)},
            #       ramdom: {custom_format(random_result)}, std: {custom_format(random_std)}, loss:{custom_format(random_loss)}, std: {custom_format(random_loss_std)}')
            # print(save_name_id, f'{len(rtn_obs)} samples avg: {result}, std: {std},loss: {loss},loss_std: {loss_std}, ramdom: {random_result}, std: {random_std}, loss:{random_loss}, std: {random_loss_std}')
            random_result_list.append(random_result)
            random_std_list.append(random_std)
            random_loss_list.append(random_loss)
            random_loss_std_list.append(random_loss_std)

        else:
            print(save_name_id, f'{set_size} {dataset_name} samples: {result}, std: {std}, loss: {loss}, std: {loss_std}')
        
        # save trained policies and statistics
        cpu_state_dict = {k: v.cpu() for k, v in saved_param.items()}
        policy_list.append(cpu_state_dict)
        
        result_list.append(result)
        std_list.append(std)
        loss_list.append(loss)
        loss_std_list.append(loss_std)

        if rounds+1 == args.rounds:
            eval.close()
            id_list = list(range(args.rounds))

            result_array_datasize_list = np.array(id_list) * args.data_per_round + args.data_per_round 

            result_array = np.array(result_list)
            rollout_std_array = np.array(std_list)
            loss_array = np.array(loss_list)
            loss_std_array = np.array(loss_std_list)
            final_mean_result = [result_array,rollout_std_array,loss_array,loss_std_array]

            random_result_array = np.array(random_result_list)
            random_rollout_std_array = np.array(random_std_list)
            random_loss_array = np.array(random_loss_list)
            random_loss_std_array = np.array(random_loss_std_list)
            final_random_result = [random_result_array,random_rollout_std_array,random_loss_array,random_loss_std_array]


            return result_array_datasize_list, policy_list,final_mean_result,final_random_result


# if __name__ == "__main__":
    


