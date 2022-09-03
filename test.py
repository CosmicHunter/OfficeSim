from officesimenv import OfficeEnv
import gym
import pygame
from collections import defaultdict

env = OfficeEnv()

episodes = 100
num_agents = 2
flag = True
faulty_count = 0

for episode in range(episodes):
	state_dict = defaultdict(list)
	reward_dict = defaultdict(list)
	print("Episode :  ",episode)
	done = [False] * num_agents
	al = [None] * num_agents
	obs = env.reset()
	if env.temp_chk() == False:
		faulty_count+=1

	count = 0
	while count < 1:
		env.update_last_state(obs)
		obs, rewards, dones, infos = env.step(al)
		# for idx in range(len(env.agent_list)):
			# state_dict[idx].append(env.agent_list[idx].last_state)
			# reward_dict[idx].append(rewards[idx])
		count+=1
	env.render()
print("faulty_test cases : " , faulty_count)	

print("total simulation steps : ",env.simulation_step)