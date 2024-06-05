import numpy as np
import torch
import gym

from dril.a2c_ppo_acktr import utils
from dril.a2c_ppo_acktr.envs import make_vec_envs
import torch.nn.functional as F

class eval_ensemble_class:
    def __init__(self, ensemble_size, ob_rms, env_name, seed, num_processes, eval_log_dir,
        device, num_episodes=None, stats_path=None, hyperparams=None, time=False):
        super(eval_ensemble_class, self).__init__()

        self.num_processes = num_processes
        self.num_episodes = num_episodes
        self.device = device
        self.ensemble_size = ensemble_size
        self.eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, 0.99, eval_log_dir, device,\
                    True, stats_path=stats_path, hyperparams=hyperparams, time=time)
        self.eval_envs.reset()

    def test(self,ensemble_policy,expert,random_selection):
        step=0
        eval_episode_rewards = []
        eval_episode_loss = []
        obs = self.eval_envs.reset()
        current_episode_rewards = np.zeros(self.num_processes)
        current_episode_expert_loss = np.zeros(self.num_processes)  
        while len(eval_episode_rewards) < self.num_episodes:
            selected_actions = torch.zeros((obs.shape[0],self.eval_envs.action_space.low.shape[0])).to(self.device)
            with torch.no_grad():
                ensemble_obs = torch.unsqueeze(obs, dim=0)
                ensemble_obs = torch.cat([ensemble_obs.repeat(self.ensemble_size, *[1]*len(ensemble_obs.shape[1:]))], dim=0)
                ensemble_actions = ensemble_policy(ensemble_obs)
                if random_selection:
                    for i in range(obs.shape[0]):
                        selected_actions[i] = ensemble_actions[torch.randint(low=0, high=self.ensemble_size, size=())][i]   
                else:
                    selected_actions = ensemble_actions 
                # get expert action for loss
                _, expert_action, _, _ = expert.act(obs, None, None, deterministic=True)
            if isinstance(self.eval_envs.action_space, gym.spaces.Box):
                clip_ensemble_actions = torch.clamp(selected_actions, float(self.eval_envs.action_space.low[0]),\
                            float(self.eval_envs.action_space.high[0])) 
                clip_expert_action = torch.clamp(expert_action, float(self.eval_envs.action_space.low[0]),\
                            float(self.eval_envs.action_space.high[0])) 
            else:
                clip_ensemble_actions = selected_actions
                clip_expert_action = expert_action
            
            if not random_selection:
                clip_ensemble_actions = torch.mean(clip_ensemble_actions, dim=0)

            obs, reward, done, _ = self.eval_envs.step(clip_ensemble_actions)

            # this counting steps may lead to bad implementation
            step += 1

            current_episode_rewards += reward.cpu().numpy().flatten()
            squared_diffs = ((clip_ensemble_actions - clip_expert_action) ** 2).mean(dim=1)
            current_episode_expert_loss += squared_diffs.cpu().numpy()
            # Check if episodes are done
            for i, done_ in enumerate(done):
                if done_:
                    eval_episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0  
                    eval_episode_loss.append(current_episode_expert_loss[i]/step)
                    current_episode_expert_loss[i] = 0
                    # print(step)
                    # print(squared_diffs.cpu().numpy())


            if len(eval_episode_rewards) >= self.num_episodes:
                break

        return np.mean(eval_episode_rewards),np.std(eval_episode_rewards),np.mean(eval_episode_loss),np.std(eval_episode_loss)
    
    def close(self):
        self.eval_envs.close()





def eval_ensemble(ensemble_policy, ensemble_size, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, num_episodes=None, stats_path=None, hyperparams=None, time=False):
    # eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
    #                           None, eval_log_dir, device, True, atari_max_steps)
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, 0.99, eval_log_dir, device,\
                    True, stats_path=stats_path, hyperparams=hyperparams, time=time)
    # print('eval')
    # print('num_processes',num_processes)
    # vec_norm = utils.get_vec_normalize(eval_envs)
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     vec_norm.ob_rms = ob_rms
    eval_episode_rewards = []
    # print('eval0') reset gets the argv[0]=  outout
    # print('init')
    obs = eval_envs.reset()
    # print('init done')
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.zeros(num_processes, 1, device=device)
    # print(num_episodes)
    current_episode_rewards = np.zeros(num_processes) 
    while len(eval_episode_rewards) < num_episodes:
        selected_actions = torch.zeros((obs.shape[0],eval_envs.action_space.low.shape[0])).to(device)
        with torch.no_grad():
            ensemble_obs = torch.unsqueeze(obs, dim=0)
            ensemble_obs = torch.cat([ensemble_obs.repeat(ensemble_size, *[1]*len(ensemble_obs.shape[1:]))], dim=0)
            ensemble_actions = ensemble_policy(ensemble_obs)
            for i in range(obs.shape[0]):
                selected_actions[i] = ensemble_actions[torch.randint(low=0, high=ensemble_size, size=())][i]   
        if isinstance(eval_envs.action_space, gym.spaces.Box):
            clip_ensemble_actions = torch.clamp(selected_actions, float(eval_envs.action_space.low[0]),\
                         float(eval_envs.action_space.high[0])) 
        else:
            clip_ensemble_actions = selected_actions

        obs, reward, done, infos = eval_envs.step(clip_ensemble_actions)
        
        current_episode_rewards += reward.cpu().numpy().flatten()

        # Check if episodes are done
        for i, done_ in enumerate(done):
            if done_:
                eval_episode_rewards.append(current_episode_rewards[i])
                # print('done',current_episode_rewards[i],num_episodes)
                # Reset the reward accumulator 
                current_episode_rewards[i] = 0  

        if len(eval_episode_rewards) >= num_episodes:
            # print('gather enouth trails',num_episodes)
            break

    eval_envs.close()

    # while len(eval_episode_rewards) < num_episodes:
    #     selected_actions = torch.zeros((obs.shape[0],eval_envs.action_space.low.shape[0])).to(device)
    #     with torch.no_grad():
    #         # _, action, _, eval_recurrent_hidden_states = actor_critic.act(
    #         #     obs,
    #         #     eval_recurrent_hidden_states,
    #         #     eval_masks,
    #         #     deterministic=True)
    #         # print(obs.shape)
    #         ensemble_obs = torch.unsqueeze(obs, dim=0)
    #         ensemble_obs = torch.cat([ensemble_obs.repeat(ensemble_size, *[1]*len(ensemble_obs.shape[1:]))], dim=0)
    #         ensemble_actions = ensemble_policy(ensemble_obs)
    #         # .view(ensemble_size, -1)
    #         for i in range(obs.shape[0]):
    #             selected_actions[i] = ensemble_actions[torch.randint(low=0, high=ensemble_size, size=())][i]   
    #             # selected_actions[i] = ensemble_actions[np.random.randint(low=0, high=ensemble_size)][i]            # 
    #     # print(obs.shape[0],eval_envs.action_space.low.shape[0])
    #     # print(selected_action)
    #     # Obser reward and next obs
    #     if isinstance(eval_envs.action_space, gym.spaces.Box):
    #         # clip_action = torch.clamp(action, float(eval_envs.action_space.low[0]),\
    #         #              float(eval_envs.action_space.high[0]))         
    #         # clip_ensemble_actions = np.clip(selected_actions, eval_envs.action_space.low, eval_envs.action_space.high)
    #         clip_ensemble_actions = torch.clamp(selected_actions, float(eval_envs.action_space.low[0]),\
    #                      float(eval_envs.action_space.high[0])) 
    #     else:
    #         # clip_action = action
    #         clip_ensemble_actions = selected_actions
            
    #     # Obser reward and next obs
    #     obs, _, done, infos = eval_envs.step(clip_ensemble_actions)
    #     # obs, _, done, infos = eval_envs.step(clip_action)
    #     eval_masks = torch.tensor(
    #         [[0.0] if done_ else [1.0] for done_ in done],
    #         dtype=torch.float32,
    #         device=device)

    #     for info in infos:
    #         if 'episode' in info.keys():
    #             eval_episode_rewards.append(info['episode']['r'])
    # eval_envs.close()

    # print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
    #     len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    
    # print(np.mean(eval_episode_rewards), np.std(eval_episode_rewards))
    return np.mean(eval_episode_rewards),np.std(eval_episode_rewards)
