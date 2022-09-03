import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from explorer import Explorer
import time
from officesimenv import OfficeEnv
import gym
import pygame
from newtestpolicy1 import *
from policyfactory import *

env = OfficeEnv()
# in order to visualize at each step of test case we use this script
# python visualizer_testing.py --model_dir data/output --phase test --policy newtestpolicy1/internal --test_case 0
def main():
	parser = argparse.ArgumentParser('Parse configuration file')
	parser.add_argument('--env_config', type=str, default='configs/env.config')
	parser.add_argument('--policy_config', type=str, default='configs/policy.config')
	parser.add_argument('--policy', type=str, default=None)
	parser.add_argument('--model_dir', type=str, default=None)
	parser.add_argument('--il', default=False, action='store_true')
	parser.add_argument('--gpu', default=False, action='store_true')
	parser.add_argument('--phase', type=str, default='test')
	parser.add_argument('--test_case', type=int, default=None)
	args = parser.parse_args()

	if args.model_dir is not None:
		env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
		policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
		if args.il:
			model_weights = os.path.join(args.model_dir, 'il_model.pth')
		else:
			if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
				model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
				
			else:
				# model_weights = os.path.join(args.model_dir, 'rl_model.pth')
				model_weights = os.path.join(args.model_dir, 'ann_from_il_model.pth')
			
	else:
		env_config_file = args.env_config
		policy_config_file = args.policy_config
	# configure logging and device
	logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
						datefmt="%Y-%m-%d %H:%M:%S")
	device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
	logging.info('Using device: %s', device)
	
	# configure Environment
	env_config = configparser.RawConfigParser()
	env_config.read(env_config_file)
	env.configure(env_config)
	num_agents = env_config.getint('sim','num_agent')
	# configure policy
	if args.policy == 'newtestpolicy1' or args.policy == 'internal':
		policy = newtestpolicy1()
	policy_config = configparser.RawConfigParser()
	policy_config.read(policy_config_file)
	policy.time_step = env.time_step
	policy.env = env
	# if args.policy == 'newtestpolicy1':
	policy.configure_other_params(num_agents)
	print("After Configuring Num agents for policy", policy.num_agents)
	print("Other Agents After Configuring other params " , policy.other_agents)
	print("policy config : " , policy_config_file)
	policy.configure(policy_config)
	if policy.trainable and args.policy != None and args.policy != 'internal':
		if args.model_dir is None:
			parser.error('Trainable policy must be specified with a model weights directory')
		print("policy :" , policy)
		print("Model policy : " , policy.get_model())
		policy.get_model().load_state_dict(torch.load(model_weights,map_location=torch.device('cpu')))
	
	# explorer = Explorer(env, agent_list, device, gamma=0.9)
	explorer = Explorer(env , device ,num_agents, gamma = 0.9,target_policy = policy)
	policy.set_phase(args.phase)
	policy.set_device(device)
	
	


	isRunning = True
	move_flag = False
	# step = 0
	obs = env.reset(args.phase , args.test_case)
	env.render()
	done_list = [False for i in range(num_agents)]      
	sc  = 0
	while isRunning:
		# env.render()
		for evt in pygame.event.get():
			if evt.type == pygame.QUIT:
				isRunning = False
				

			if evt.type == pygame.KEYDOWN:
				if evt.key == pygame.K_SPACE:
					move_flag = not(move_flag)

			if evt.type == pygame.MOUSEMOTION:
				mx , my = pygame.mouse.get_pos()
				caption = f"({mx},{my})"
				pygame.display.set_caption(caption)
				# print("mx,my : ",mx,my)
			  
			  
		fa =[]
		if move_flag:
			env.update_last_state(obs)
			if args.policy == 'newtestpolicy1':
				action_list = []
			else:
				action_list = [None] * num_agents
				
			
			if args.policy == 'newtestpolicy1':
				for i in range(len(env.agent_list)):
					if done_list[i] == True:
						action_list.append(ActionXY(0,0))
						
					else:        
						action_list.append(env.agent_list[i].act(env.agent_list[i].last_state , policy))
					
			obs, rew, done_list, info_list , _ = env.step(action_list , render = True)
			for i, agent in enumerate(env.agent_list):
				print(f"Agent {i} : action : {agent.action}")
			print(f"STEP : {sc} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
			print(f"step : {sc} , info list : {info_list}")
			print("Rewards : ",rew)

			sc+=1
			# if env.is_done(done_list):
				# move_flag = False
		if not(isRunning):
			break
	print("num steps",env.simulation_step)
	env.close()       

	# obs = env.reset(args.phase , args.test_case)
	# done = False
	# done_list = [False for i in range(num_agents)]
	# # last_pos = np.array(robot.get_position())
	# sc = 0
	# while not env.is_done(done_list):
	# 	t1 = time.time()
	# 	env.update_last_state(obs)
	# 	if args.policy == 'newtestpolicy1':
	# 		action_list = []
	# 	else:
	# 		action_list = [None] * num_agents
			
	# 	t1 = time.time()

	# 	if args.policy == 'newtestpolicy1':
	# 		for i in range(len(env.agent_list)):
	# 			if done_list[i] == True:
	# 				action_list.append(ActionXY(0,0))
					
	# 			else:        
	# 				action_list.append(env.agent_list[i].act(env.agent_list[i].last_state , policy))
	# 	t2 = time.time()
	# 	print(f"STEP : {sc} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
	# 	print("collect action time : ",t2 - t1 )
	# 	# print("predict call cnt " , agent_list[-1].policy.pcnt)

	# 	obs, rew, done_list, info_list , _ = env.step(action_list)
	# 	print(f"step : {sc} , rewards : {rew}")
	# 	t3=  time.time()
	# 	print(f"step : {sc} , info list : {info_list}")
	# 	print("environment step time :" , t3 - t2)
	# 	sc+=1
		
	# 	# current_pos = np.array(robot.get_position())
	# 	# logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
	# 	# last_pos = current_pos
	
	# env.render()
	# print(sc)

if __name__ == '__main__':
	main()
