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

# python state_action_test.py --model_dir data/output --phase test --policy newtestpolicy1/internal --visualize --test_case 0
# python state_action_test.py --model_dir data/output --phase test --policy newtestpolicy1 --visualize --test_case 0

def main():
	parser = argparse.ArgumentParser('Parse configuration file')
	parser.add_argument('--env_config', type=str, default='configs/env.config')
	parser.add_argument('--policy_config', type=str, default='configs/policy.config')
	parser.add_argument('--policy', type=str, default=None)
	parser.add_argument('--model_dir', type=str, default=None)
	parser.add_argument('--il', default=False, action='store_true')
	parser.add_argument('--gpu', default=False, action='store_true')
	parser.add_argument('--visualize', default=False, action='store_true')
	parser.add_argument('--phase', type=str, default='test')
	parser.add_argument('--test_case', type=int, default=None)
	args = parser.parse_args()

	if args.model_dir is not None:
		env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
		policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
		if args.il:
			model_weights = os.path.join(args.model_dir, 'il_model.pth')
		else:
			if os.path.exists(os.path.join(args.model_dir, 'resumed_state_action_model.pth')):
				model_weights = os.path.join(args.model_dir, 'resumed_state_action_model.pth')
			else:
				# model_weights = os.path.join(args.model_dir, 'rl_model.pth')
				model_weights = os.path.join(args.model_dir, 'state_action_gcn_model.pth')
				print("loaded state action model weights ")
	else:
		# print("yeloooo")
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
		print("model loaded to policy")
		print(policy.get_model())
	# explorer = Explorer(env, agent_list, device, gamma=0.9)
	explorer = Explorer(env , device ,num_agents, gamma = 0.9,target_policy = policy)
	policy.set_phase(args.phase)
	policy.set_device(device)
	
	if args.visualize:
		obs = env.reset(args.phase , args.test_case)
		done = False
		done_list = [False for i in range(num_agents)]
		# last_pos = np.array(robot.get_position())
		sc = 0
		while not env.is_done(done_list):
			t1 = time.time()
			env.update_last_state(obs)
			if args.policy == 'newtestpolicy1':
				action_list = []
			else:
				action_list = [None] * num_agents
				
			t1 = time.time()

			if args.policy == 'newtestpolicy1':
				for i in range(len(env.agent_list)):
					if done_list[i] == True:
						action_list.append(ActionXY(0,0))
						
					else:        
						action_list.append(env.agent_list[i].get_action(env.agent_list[i].last_state , policy))
			# print(action_list)
			t2 = time.time()
			print(f"STEP : {sc} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
			print("collect action time : ",t2 - t1 )
			# print("predict call cnt " , agent_list[-1].policy.pcnt)

			obs, rew, done_list, info_list , _ = env.step(action_list)
			# print("observations : ",obs)
			# print(f"step : {sc} , rewards : {rew}")
			t3=  time.time()
			print(f"step : {sc} , info list : {info_list}")
			print("environment step time :" , t3 - t2)
			sc+=1
			
			# current_pos = np.array(robot.get_position())
			# logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
			# last_pos = current_pos
		
		env.render()
		print(sc)
	else:
		print("Inside Test Code without Visualization")
		print(f"Running explorer for {env.case_size[args.phase]}  number of episodes")

		explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
		print(f"Testing on {env.case_size[args.phase]} Test Cases")


if __name__ == '__main__':
	main()