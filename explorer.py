import logging
import copy
import torch
from officesimenv import OfficeEnv
import gym
import pygame
from collections import defaultdict
from info import *
from pipcheck_modified import *
from helper_functions import average
import time
import os 
import numpy as np 

# explorer time too long
class Explorer():   
    def __init__(self, env, device, num_agents,memory=None, gamma=None, target_policy=None):
        self.env = env
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.robot_v_pref = 1
        self.robot_time_step = 0.25
        self.all_in_one_grp = False
        self.avg_convex_hull_violations = 0
        self.traj_with_violation = 0
        self.group_breach_violation = 0
        self.num_agents = num_agents
        self.wall_collisions_per_agent_per_ep = 0
        print(f"number of agents inside explorer is {self.num_agents}")
    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # flags 
    # state_action_method : if true stores state , action in memory instead of state , value
    # gen txt data : if true creates text data for trajectory prediction
    # for gen txt data we need gen_data folder
    # gen state traj data : is for generating sequential states trajectory data that can be 
    # stored in memory for sequence prediction.

    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, ep=None,
                       print_failure=False , state_action_method = False , gen_txt_data = False , gen_state_traj_data = False):
        faulty_count = 0
        episodes = k
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards_dict = defaultdict(list)
        collision_cases = []
        timeout_cases = []
        success_cases = []
        cases_with_hull_violation = []
        self.traj_with_violation = 0
        self.group_breach_violation = 0
        self.avg_convex_hull_violations = 0

        self.avg_spf_rew_per_agent = defaultdict(int)
        self.spf_violations = defaultdict(int)
        self.non_violations = defaultdict(int)
        self.wall_collisions_per_agent_per_ep = 0
        self.target_policy.set_phase(phase)
        prev_s=0
        if gen_txt_data == True:
            episode_traj_data = []
            time_counter = 0
        for episode in range(episodes):
            if imitation_learning:
                print(f"il episode : {episode}")
            
            state_dict = defaultdict(list)
            reward_dict = defaultdict(list)
            action_dict = defaultdict(list)
            # print("Episode :  ",episode)
            dones = [False]* self.num_agents
            obs = self.env.reset(phase)
            if gen_txt_data == True:
                time_counter += 1
                for idx,agent in enumerate(self.env.agent_list):
                    episode_traj_data = episode_traj_data + [time_counter,idx+episode*self.num_agents,agent.x,agent.y]
                print("time counter : ",time_counter)
                if episode == 4:
                    episode_traj_data = np.reshape(episode_traj_data,(-1,4))
                    with open('gen_data/gen_test_data.txt','wb') as f:
                        np.savetxt(f, episode_traj_data, fmt=('%.1f %.1f %.10f %.10f'))
                    episode_traj_data = []
                if episode == 24:
                    episode_traj_data = np.reshape(episode_traj_data,(-1,4))
                    with open('gen_data/gen_val_data.txt','wb') as f:
                        np.savetxt(f, episode_traj_data, fmt=('%.1f %.1f %.10f %.10f'))
                    episode_traj_data = []

            if self.env.temp_chk() == False:
                faulty_count+=1

            count = 0
            t1 = time.time()
            while not(self.env.is_done(dones)):
                t1 = time.time()
                self.env.update_last_state(obs)
                al = [None] * self.num_agents
                if imitation_learning == False:
                    al = []
                    for idx in range(len(self.env.agent_list)):
                        al.append(self.env.agent_list[idx].act(self.env.agent_list[idx].last_state, self.target_policy))
                obs, rewards, dones, infos , spf_rew_data = self.env.step(al)
                
                if gen_txt_data == True:
                    time_counter += 1
                    for idx,agent in enumerate(self.env.agent_list):
                        episode_traj_data = episode_traj_data + [time_counter,idx+episode*self.num_agents,agent.x,agent.y]

                # print("rewards recieved for this step :" , rewards)
                for idx in range(len(self.env.agent_list)):
                    if imitation_learning == False:
                        state_dict[idx].append(self.target_policy.last_state[idx])
                    else:
                        state_dict[idx].append(self.env.agent_list[idx].last_state)
                    reward_dict[idx].append(rewards[idx])

                    if imitation_learning and state_action_method:
                        action_dict[idx].append(self.env.agent_list[idx].action)
                count+=1

                # convex hull penetration check
                if violation_check(self.env.agent_list[0].last_state) == True:
                    self.group_breach_violation+=1

                for idx in range(len(self.env.agent_list)):
                    srew , flag = spf_rew_data[idx]
                    self.avg_spf_rew_per_agent[idx] += srew
                    if flag == -1:
                        self.spf_violations[idx] += 1
                    elif flag == 1:
                        self.non_violations[idx] += 1

                for idx in range(len(self.env.agent_list)):
                    if isinstance(infos[idx], Danger):
                        too_close += 1
                        min_dist.append(infos[idx].min_dist)

                for idx in range(len(self.env.agent_list)):
                    if infos[idx] == "wallcollision":
                        self.wall_collisions_per_agent_per_ep+=1

                t2 = time.time()
                # print("time for one step : ",t2 - t1)        
                        
            for idx in range(len(self.env.agent_list)):
                info = infos[idx]
                if info == "goalreached":
                    success += 1
                    success_times.append(self.env.global_time)
                elif info == "collision" or info == "wallcollision":
                    collision += 1
                    collision_cases.append(episode)
                    collision_times.append(self.env.global_time)
                    # update done list and break out of loop
                elif info == "timeout":
                    timeout += 1
                    timeout_cases.append(episode)
                    timeout_times.append(self.env.time_limit)
                elif isinstance(info , GroupCollided):
                    pass
                else:
                    print(info)
                    raise ValueError('Invalid end signal from environment')

            if (success-prev_s) == self.num_agents:
                success_cases.append(episode)
            prev_s = success
            t3 = time.time()
            
            


            if update_memory:
                for idx in range(len(self.env.agent_list)):
                    info = infos[idx]
                    if info == "goalreached" or info == "collision" or info =="wallcollision":
                        # only add positive(success) or negative(collision) experience in experience set
                        # t5 = time.time()
                        if state_action_method:
                            print("state action method update intitated !!")
                            self.update_memory(state_dict[idx], reward_dict[idx], action_dict[idx],imitation_learning , state_action_method)
                        else:
                            print(" not state action not initated update")
                            self.update_memory(state_dict[idx], reward_dict[idx], imitation_learning)

                if gen_state_traj_data:
                    print("state trajectory data !! block")
                    self.update_memory_state_traj(state_dict, imitation_learning) 
                        
            t4 = time.time()
            print("time to update memory : ", t4 - t3)
            if self.group_breach_violation > 0:
                self.traj_with_violation+=1
                cases_with_hull_violation.append(episode)
                self.avg_convex_hull_violations += self.group_breach_violation

            for idx in range(len(self.env.agent_list)):            
                cumulative_rewards_dict[idx].append(sum([pow(self.gamma, t * self.env.agent_list[idx].time_step * self.env.agent_list[idx].v_pref)
                                           * reward for t, reward in enumerate(reward_dict[idx])]))    # do it for all agents
        

        # out of the episode for loop
        avg_cumulative_rewards = []
        for j in range(len(cumulative_rewards_dict[0])):
            temp = 0
            for i in range(len(self.env.agent_list)):
                temp += cumulative_rewards_dict[i][j]
            avg_cumulative_rewards.append(temp / len(self.env.agent_list))

        if gen_txt_data:
            episode_traj_data = np.reshape(episode_traj_data,(-1,4))
            with open('gen_data/gen_train_data.txt','wb') as f:
                np.savetxt(f, episode_traj_data, fmt=('%.1f %.1f %.10f %.10f'))
            
        success_rate = success / k
        print("success : ",success)
        print("timeouts : ",timeout)
        collision_rate = collision / k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit
        self.avg_convex_hull_violations/=k
        self.wall_collisions_per_agent_per_ep /= k
        self.wall_collisions_per_agent_per_ep /= self.num_agents

        for key , v in self.avg_spf_rew_per_agent.items():
            self.avg_spf_rew_per_agent[key] = v/k

        for key , v in self.spf_violations.items():
            self.spf_violations[key] = v/k

        for key , v in self.non_violations.items():
            self.non_violations[key] = v/k


        extra_info = '' if ep is None else 'in episode {} '.format(ep)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(avg_cumulative_rewards)))
       
        # if self.all_in_one_grp == False:
        #     logging.info(f'total reward for group 1 agent is {average(cumulative_rewards_dict[len(self.agent_list)-1])}')
       
        if phase == 'train':
            logging.info(f'Average Number of Time violation of group hull per trajectory : {self.avg_convex_hull_violations}')

        elif phase == 'val' or phase =='test' or imitation_learning == True:
            logging.info(f'Average Number of Violation of Convex hull per trajectory : {self.avg_convex_hull_violations}')
            logging.info(f'Number of Trajectories in which violation was encountered : {self.traj_with_violation}')
        
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot_time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        

        if print_failure:
            collision_cases = list(set(collision_cases))
            timeout_cases = list(set(timeout_cases))
            logging.info(f'Collision cases: ({len(collision_cases)}) : ' + ' '.join([str(x) for x in collision_cases]))
            logging.info(f'Timeout cases: ({len(timeout_cases)}) : ' + ' '.join([str(x) for x in timeout_cases]))
            logging.info(f'Complete Success cases : ({len(success_cases)}) , : ' + ' '.join([str(x) for x in success_cases]))
            # logging.info(f'Group Reach cases : ({len(grp_reach_cases)}) , : ' + ' '.join([str(x) for x in grp_reach_cases]))
            # logging.info(f'Only Grp1 reach cases : ({len(only_grp1_success_cases)}) ' + ' '.join([str(x) for x in only_grp1_success_cases]))
            logging.info(f'Convex Hull Violation Cases : ({len(cases_with_hull_violation)}) ' + ' '.join([str(x) for x in cases_with_hull_violation]))
       
        

    def update_memory(self, states, rewards, actions=None,imitation_learning=False,state_action_method=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)  # here the states are coming from orca policy , so they are joint states , hence we transform them
                # Size of state here [39]
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot_time_step * self.robot_v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    # print("Inside inu else part " , next_state.size())  # size is [3 , 13] transformed state
                    gamma_bar = pow(self.gamma, self.robot_time_step * self.robot_v_pref)
                    # print("RL PART in inside explorer update_memory  , next state -> ",next_state)
                    # print("next_state.unsqueeze(0)" , next_state.unsqueeze(0))
                    next_state_normalized = normalized_inputs = nn.functional.normalize(next_state.unsqueeze(0), p=2.0, dim=1, eps=1e-12, out=None)
                    value = reward + gamma_bar * self.target_model(next_state_normalized).data.item()
            
            if state_action_method:
                # print("state action memory update !")
                action = actions[i]
                action = torch.Tensor([action.vx , action.vy]).to(self.device)
                self.memory.push((state , action))
                # print("pushed this into mem : ",state , action)
            else:
                value = torch.Tensor([value]).to(self.device)   
                self.memory.push((state, value))


    # this method is for updating memory in case of state trajectory sequence data
    # where target is the 5th positon while the input would be the sequence tensor of states
    def update_memory_state_traj(self, state_dict,imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        states = state_dict[0]
        tranformed_states_per_frame = []
        agent_cords_per_frame = []
        for i, _ in enumerate(states):
            
            if imitation_learning:
                for idx in range(len(self.env.agent_list)):
                    state = self.target_policy.transform(state_dict[idx][i])  # here the states are coming from orca policy , so they are joint states , hence we transform them
                    tranformed_states_per_frame.append(state)
                    agent_cords_per_frame.append(self.env.agent_list[idx].getPos())
            
            self.memory.push((tranformed_states_per_frame[0],tranformed_states_per_frame[1],agent_cords_per_frame[0],agent_cords_per_frame[1]))
            tranformed_states_per_frame = []
            agent_cords_per_frame = []
                