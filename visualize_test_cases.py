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

# this script is for visualizing various test case 
python visualize_test_cases.py --model_dir data/output/ --policy newtestpolicy1/internal --visualize --num_cases 5


def main():
	parser = argparse.ArgumentParser('Parse configuration file')
	parser.add_argument('--env_config', type=str, default='configs/env.config')
	parser.add_argument('--policy_config', type=str, default='configs/policy.config')
	parser.add_argument('--model_dir', type=str, default=None)
	parser.add_argument('--il', default=False, action='store_true')
	parser.add_argument('--gpu', default=False, action='store_true')
	parser.add_argument('--visualize', default=False, action='store_true')
	parser.add_argument('--phase', type=str, default='test')
	parser.add_argument('--test_case', type=int, default=None)
	parser.add_argument('--num_cases',type = int , default = None)
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
				model_weights = os.path.join(args.model_dir, 'rl_model.pth')
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
	print("num ",num_agents)
	num_cases = args.num_cases
	explorer = Explorer(env , device ,num_agents, gamma = 0.9,target_policy = None)
	if num_cases == None:
		num_cases = 1
	if args.visualize:
		for case in range(num_cases):
			if args.phase == 'test':
				obs = env.reset(args.phase , case)
			else:
				obs = env.reset(args.phase , args.test_case)
			done_list = [False for i in range(num_agents)]
			sc = 0
			while sc < 2:
				t1 = time.time()
				env.update_last_state(obs)
				action_list = [None] * num_agents
				print(f"STEP : {sc} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
				# print("predict call cnt " , agent_list[-1].policy.pcnt)

				obs, rew, done_list, info_list,_ = env.step(action_list)
				print(f"step : {sc} , rewards : {rew}")
				t2=  time.time()
				print(f"step : {sc} , info list : {info_list}")
				print("time :" , t2 - t1)
				sc+=1
				# current_pos = np.array(robot.get_position())
				# logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
				# last_pos = current_pos
			
			env.render()
			print(sc)
		

if __name__ == '__main__':
	main()