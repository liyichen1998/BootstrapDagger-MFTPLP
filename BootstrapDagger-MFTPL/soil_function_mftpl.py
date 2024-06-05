import gym, os
import numpy as np
import argparse
import random
import pandas as pd
import copy
import pickle
import time

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
    'Walker2DBulletEnv-v0': 24   
    }

def custom_format(num):
    if num >= 1:
        return f"{num:.6f}"
    else:
        return f"{num:.3e}"

def soil_function(algorithm,ensemble_size,env_name,device,seed =1,random_selection=True,non_realizable=False,linear=False,noisy_expert=False,additional_sample_rate=1,additional_sample_name="dagger_1.pkl",additional_data_folder = 'noisy_expert_rep_result',rl_baseline_zoo_dir='xxxx/rl-baselines-zoo'):
    args = get_args()

    args.behavior_cloning = True if algorithm=='bc' else False
    if  algorithm=='logger':
        args.ensemble_shuffle_type = 'bootstrap'
    elif algorithm=='elogger':  #original dataset + bootstrap
        args.ensemble_shuffle_type = 'enhance_bootstrap'
    elif algorithm=='mftpl':  #original dataset + additional samples 
        args.ensemble_shuffle_type = 'enhance_sample'   
    elif algorithm=='dmftpl':  #original dataset + dynamically assigned size of additional samples sqrt(n*sample_per_round)
        args.ensemble_shuffle_type = 'dynamic_enhance_sample' 
    else:
        args.ensemble_shuffle_type = 'norm_shuffle'

    # if algorithm=='elogger':
    #     args.train_epoch = int(args.train_epoch /2) 

    # args.ensemble_shuffle_type = 'bootstrap' if algorithm=='logger' else 'norm_shuffle'

    args.ensemble_size = ensemble_size
    args.device = device
    if non_realizable:
        args.rounds = 40 if env_name in ['AntBulletEnv-v0','HopperBulletEnv-v0'] else 50
    else:
        args.rounds = 20 if env_name in ['AntBulletEnv-v0','HopperBulletEnv-v0'] else 50
    args.random_selection =random_selection
    args.non_realizable = non_realizable
    args.linear = linear
    args.noisy_expert = noisy_expert
    args.seed = seed

    print('Env Name: ',env_name, 'rounds: ', args.rounds)
    print('Use Noisy Expert: ', noisy_expert)
    print('Use Random Ensemble Selections for Data Gathering: ',random_selection)

    if args.non_realizable:
        args.hidden_size = env_to_hiddensize_dict[env_name]


    if args.behavior_cloning:
        print("bc",args.ensemble_size)
        save_name_id = "bc_"+str(args.ensemble_size)
    elif args.ensemble_shuffle_type=='norm_shuffle':
        print("dagger",args.ensemble_size)
        save_name_id = "dagger_"+str(args.ensemble_size)
    elif args.ensemble_shuffle_type=='enhance_bootstrap': #original dataset + bootstrap
        print("elogger",args.ensemble_size)
        save_name_id = "elogger_"+str(args.ensemble_size)
    elif args.ensemble_shuffle_type=='enhance_sample': #original dataset + samples from dagger dataset
        print("mftpl",args.ensemble_size)
        save_name_id = "mftpl"+str(args.ensemble_size)  
    elif args.ensemble_shuffle_type=='dynamic_enhance_sample': #original dataset + samples from dagger dataset
        print("dynamic_mftpl with rate",args.ensemble_size)
        save_name_id = "mftpl"+str(args.ensemble_size)    
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
    
    # loading additional data
    if args.ensemble_shuffle_type in ['enhance_sample','dynamic_enhance_sample']: #original dataset + samples from dagger dataset
        print("loading additional data")
        additional_data_path = os.path.join(os.getcwd(), additional_data_folder, env_name,additional_sample_name)
        if not os.path.exists(additional_data_path):
            print('additional data not found:',additional_data_path)
            sys.exit()
        else:
            # load data
            with open(additional_data_path, 'rb') as f:
                results_dict = pickle.load(f)
                obs_data = results_dict['concatenated_obs']
                obs_data = obs_data.reshape(-1, obs_data.shape[-1])
                acs_data = results_dict['concatenated_acs']
                acs_data = acs_data.reshape(-1, acs_data.shape[-1])
            # decide perturbation size
            if args.ensemble_shuffle_type == 'enhance_sample':
                additional_sample_budget = int(args.rounds * args.data_per_round / additional_sample_rate)
                print(additional_sample_budget, 'additional data per training')
            else:
                additional_sample_budget = 0
                # int(additional_sample_rate*np.sqrt(args.data_per_round))


    rtn_obs, rtn_acs, rtn_lens, ep_rewards = [], [], [], []
    obs = env.reset()
    if args.env_name in ['duckietown']:
        obs = torch.FloatTensor([obs])

    save = True
    # print(f'[running]')

    step = 0
    # args.seed = args.seed
    idx = random.randint(1,args.subsample_frequency)

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
            if args.linear:
                print('Using naive linear models!!')
                policy_def = ensemble_models.PolicyEnsembleMLP_linear
            else:
                policy_def = ensemble_models.PolicyEnsembleMLP_nonrealizable
        else:
            policy_def = ensemble_models.PolicyEnsembleMLP_simple
        
    ensemble_policy = policy_def(*ensemble_args).to(device)

    saved_param = copy.deepcopy(ensemble_policy.state_dict())

    random_selection_list = np.random.randint(low=0, high=ensemble_size, size=1000)

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

    while True:
        with torch.no_grad():
            if args.env_name in ['highway-v0']:
                action = torch.tensor([[th_model.act(obs)]])
            elif args.env_name in ['duckietown']:
                action = torch.FloatTensor([expert.predict(None)])
            elif hasattr(gym.envs, 'atari'):
                _, actor_features, _ = th_model.base(obs, None, None)
                dist = th_model.dist(actor_features)
                action = dist.sample()
            else:
                if not args.noisy_expert:
                    _, action, _, _ = th_model.act(obs, None, None, deterministic=True)
                else:
                    _, action, _, _ = th_model.act(obs, None, None, deterministic=False)
        
            ensemble_obs = torch.unsqueeze(obs, dim=0)
            ensemble_obs = torch.cat([ensemble_obs.repeat(ensemble_size, *[1]*len(ensemble_obs.shape[1:]))], dim=0)
            ensemble_actions = ensemble_policy(ensemble_obs).squeeze(1)

        if isinstance(env.action_space, gym.spaces.Box):
            clip_action = np.clip(action.cpu(), env.action_space.low, env.action_space.high)
            clip_ensemble_actions = np.clip(ensemble_actions.cpu(), env.action_space.low, env.action_space.high)
            if args.random_selection:
                selected_action = clip_ensemble_actions[random_selection_list[step]]   
            else:
                selected_action = torch.mean(clip_ensemble_actions, dim=0)
        else:
            clip_action = action.cpu()
            clip_ensemble_actions = ensemble_actions.cpu()
            if args.random_selection:
                selected_action = clip_ensemble_actions[random_selection_list[step]]   
            else:
                selected_action = torch.mean(clip_ensemble_actions, dim=0)
                

        activate = False
        if (step == idx and args.subsample) or not args.subsample:
            rtn_obs.append(obs.cpu().numpy().copy())    #make sure the rollout have the same observation with the training
            rtn_acs.append(action.cpu().numpy().copy())
            idx += args.subsample_frequency
            activate = True
        
        if args.behavior_cloning:
            if args.env_name in ['duckietown']:
                obs, reward, done, infos = env.step(clip_action.squeeze())
                obs = torch.FloatTensor([obs])
            else:
                obs, reward, done, infos = env.step(clip_action)
        else :
            if args.env_name in ['duckietown']:
                obs, reward, done, infos = env.step(selected_action.squeeze())
                obs = torch.FloatTensor([obs])
            else:
                obs, reward, done, infos = env.step(selected_action)
        
        step += 1
        if args.env_name in ['duckietown']:
            if done:
                # print(f"reward: {reward}")
                ep_rewards.append(reward)
                save = True
                obs = env.reset()
                obs = torch.FloatTensor([obs])
                step = 0
                idx=random.randint(1,args.subsample_frequency)
                random_selection_list = np.random.randint(low=0, high=args.ensemble_size, size=1000)
                
        else:
            for info in infos or done:
                if 'episode' in info.keys():
                    # print(f"reward: {info['episode']['r']}")
                    ep_rewards.append(info['episode']['r'])
                    save = True
                    obs = env.reset()
                    step = 0
                    idx=random.randint(1,args.subsample_frequency)
                    random_selection_list = np.random.randint(low=0, high=args.ensemble_size, size=1000)
                    
        # if step!= 1 and step == idx-args.subsample_frequency+1 and len(rtn_obs) % args.data_per_round == 0:   
        if activate and len(rtn_obs) % args.data_per_round == 0:
            if int(len(rtn_obs)/args.data_per_round) in range(args.rounds+1):
                obs = env.reset()
                step = 0
                idx=random.randint(1,args.subsample_frequency)
                random_selection_list = np.random.randint(low=0, high=args.ensemble_size, size=1000)
                # print('bc',int(len(rtn_obs)/args.data_per_round))
                # print('sample size',len(rtn_obs))
                rtn_obs_ = np.concatenate(rtn_obs)
                rtn_acs_ = np.concatenate(rtn_acs)

                # ensemble_param, result, std, loss = mftpl(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_),len(rtn_obs), stats_path=stats_path, hyperparams=hyperparams, time=time )
                if args.ensemble_shuffle_type=='enhance_sample': 
                    # print(args.ensemble_shuffle_type)
                    ensemble_param = mftpl(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_), obs_data, acs_data, additional_sample_budget)
                elif args.ensemble_shuffle_type=='dynamic_enhance_sample': 
                    additional_sample_budget = int(np.sqrt(len(rtn_obs_))/additional_sample_rate)
                    # additional_sample_budget = int(len(rtn_obs_)/additional_sample_rate)
                    # print('dynamic perturbation size',additional_sample_budget)
                    ensemble_param = mftpl(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_), obs_data, acs_data, additional_sample_budget)
                else:
                    ensemble_param = mftpl(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_))
                saved_param = deepcopy(ensemble_param)
                ensemble_policy.load_state_dict(saved_param)

                result,std,loss, loss_std = eval.test(ensemble_policy,expert = th_model,random_selection=False)

                if args.ensemble_size > 1:
                    random_result,random_std,random_loss,random_loss_std = eval.test(ensemble_policy,expert = th_model,random_selection=True)
                    print(
                        save_name_id, 
                        f'{len(rtn_obs)} samples avg: {custom_format(result)}, std: {custom_format(std)},'
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
                    print(save_name_id, f'{len(rtn_obs)} samples: {result}, std: {std}, loss: {loss}, std: {loss_std}')
                
                # save trained policies and statistics
                cpu_state_dict = {k: v.cpu() for k, v in saved_param.items()}
                policy_list.append(cpu_state_dict)
                
                result_list.append(result)
                std_list.append(std)
                loss_list.append(loss)
                loss_std_list.append(loss_std)


                
            

                if int(len(rtn_obs)/args.data_per_round) % args.rounds == 0:
                    eval.close()
                    id_list = list(range(args.rounds))

                    # print('round, policy return, std across rollouts')
                    # for i in range(len(id_list)):
                    #     print(id_list[i], bc_result_list[i],bc_rollout_std[i])

                    # Converting to np array
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


                    return result_array_datasize_list, policy_list,final_mean_result,final_random_result,rtn_obs_,rtn_acs_


if __name__ == "__main__":
    start_time = time.time()

    algorithm= 'mftpl' #'mftpl'
    ensemble_size= 25
    env_name=  'Walker2DBulletEnv-v0' 
    non_realizable = False
    linear = False 
    additional_sample_rate = 166
    #  # 'HopperBulletEnv-v0'   'AntBulletEnv-v0'   'HalfCheetahBulletEnv-v0'    'Walker2DBulletEnv-v0'  
    additional_sample_name = "dagger_1.pkl"
    # "bc_1.pkl", "dagger_1.pkl"
    device = 'cuda:0'
    datasize_list, policies, result, random_result, obs, acs = soil_function(
        algorithm=algorithm,ensemble_size=ensemble_size,env_name=  env_name ,device = device, noisy_expert=True,additional_sample_rate=additional_sample_rate, non_realizable = non_realizable,linear=linear,additional_sample_name=additional_sample_name) #,noisy_expert=False, random_selection=True,non_realizable = False
    print(result)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    print(f"Algorithm: {algorithm}")
    print(f"Ensemble size: {ensemble_size}")
    print(f"Environment name: {env_name}")
