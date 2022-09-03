# Using this module we can simply collect imitation learning  data
# and train the ANN directly.
# command :- 
# python directAnntrain.py --policy newtestpolicy1

from officesimenv import OfficeEnv
import gym
import pygame
from explorer import *
import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
from trainer import Trainer
from memory import ReplayMemory
import time
from newtestpolicy1 import *
from policyfactory import *
env = OfficeEnv()
device = "cpu"

# exp = Explorer(env , device)
# exp.run_k_episodes(10, 'train')
# print("test successful")


# custom  ann only train script

# python directAnntrain.py --policy newtestpolicy1
def main():
	parser = argparse.ArgumentParser('Parse configuration file')
	parser.add_argument('--env_config', type=str, default='configs/env.config')
	parser.add_argument('--policy', type=str, default='cadrl')
	parser.add_argument('--policy_config', type=str, default='configs/policy.config')
	parser.add_argument('--train_config', type=str, default='configs/train.config')
	parser.add_argument('--output_dir', type=str, default='data/output')
	parser.add_argument('--weights', type=str)
	parser.add_argument('--resume', default=False, action='store_true')
	parser.add_argument('--gpu', default=False, action='store_true')
	parser.add_argument('--debug', default=False, action='store_true')
	args = parser.parse_args()

	# configure paths
	make_new_dir = True
	if os.path.exists(args.output_dir):
		key = input('Output directory already exists! Overwrite the folder? (y/n)')
		if key == 'y' and not args.resume:
			shutil.rmtree(args.output_dir)
		else:
			make_new_dir = False
			args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
			args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
			args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
	
	if make_new_dir:
		os.makedirs(args.output_dir)
		shutil.copy(args.env_config, args.output_dir)
		shutil.copy(args.policy_config, args.output_dir)
		shutil.copy(args.train_config, args.output_dir)
	log_file = os.path.join(args.output_dir, 'output.log')
	il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
	rl_weight_file = os.path.join(args.output_dir, 'ann_from_il_model.pth')

	# configure logging
	mode = 'a' if args.resume else 'w'
	file_handler = logging.FileHandler(log_file, mode=mode)
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO if not args.debug else logging.DEBUG
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
	
	# repo = git.Repo(search_parent_directories=True)
	# logging.info('Current git head hash code: %s'.format(repo.head.object.hexsha))
	
	device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
	logging.info('Using device: %s', device)


	# env config
	env_config = configparser.RawConfigParser()
	env_config.read(args.env_config)
	env.configure(env_config)

	# policy config
	policy = newtestpolicy1()
	policy_config = configparser.RawConfigParser()
	policy_config.read(args.policy_config)
	policy.time_step = env.time_step
	policy.env = env
	if not policy.trainable:
		parser.error('Policy has to be trainable')
	if args.policy_config is None:
		parser.error('Policy config has to be specified for a trainable network')
	
	num_agents = env_config.getint('sim','num_agent')
	print("num_agents :" , num_agents)

	if args.policy == 'spfrl' or args.policy == 'newtestpolicy1':
	   policy.configure_other_params(num_agents)
	policy.configure(policy_config)
	policy.set_device(device)


	# training config
	if args.train_config is None:
		parser.error('Train config has to be specified for a trainable network')
	train_config = configparser.RawConfigParser()
	train_config.read(args.train_config)
	rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
	train_batches = train_config.getint('train', 'train_batches')
	train_episodes = train_config.getint('train', 'train_episodes')
	sample_episodes = train_config.getint('train', 'sample_episodes')
	target_update_interval = train_config.getint('train', 'target_update_interval')
	evaluation_interval = train_config.getint('train', 'evaluation_interval')
	capacity = train_config.getint('train', 'capacity')
	epsilon_start = train_config.getfloat('train', 'epsilon_start')
	epsilon_end = train_config.getfloat('train', 'epsilon_end')
	epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
	checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

	memory = ReplayMemory(capacity)
	model = policy.get_model()
	batch_size = train_config.getint('trainer', 'batch_size')
	print('batch size is ',batch_size)
	
	trainer = Trainer(model, memory, device, batch_size)
	explorer = Explorer(env , device ,num_agents,memory , policy.gamma , target_policy = policy)
	
	if args.resume:
		if not os.path.exists(rl_weight_file):
			logging.error('RL weights does not exist')
		model.load_state_dict(torch.load(rl_weight_file))
		rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
		logging.info('Load reinforcement learning trained weights. Resume training')
	elif os.path.exists(il_weight_file):
		model.load_state_dict(torch.load(il_weight_file))
		logging.info('Load imitation learning trained weights.')
	else:
		il_t1 = time.time()
		il_episodes = train_config.getint('imitation_learning', 'il_episodes')
		il_policy = train_config.get('imitation_learning', 'il_policy')
		il_epochs = train_config.getint('imitation_learning', 'il_epochs')
		il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
		trainer.set_learning_rate(il_learning_rate)
	   
		
		
		# print(f"IL for episodes  = {il_episodes}")
		# print("robot policy class " , robot.policy.__class__.__name__)
		# print(" policy multiagent training attribute  : " , robot.policy.multiagent_training)
		
		explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
		il_t2 = time.time()
		trainer.optimize_epoch(il_epochs)
		il_t3 = time.time()
		torch.save(model.state_dict(), il_weight_file)
		il_t4 = time.time()
		logging.info('Finish imitation learning. Weights saved.')
		logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
	
	explorer.update_target_model(model)

	
	# Imitation Learning Over

	print(".............. IMITATION LEARNING OVER ...........")
	print(".............. DATA COLLECTED ..................")

	epochs = 500
	for epoch in range(epochs):
		avg_loss = trainer.optimize_epoch(1)
		avg_loss = float(avg_loss)
		if epoch != 0 and epoch % 100 == 0:
			print(f"Checkpoint Interval Reached at epoch : {epoch}")
			print("Saving Model Weights")
			torch.save(model.state_dict(), rl_weight_file)
		logging.info(f"Epoch: {epoch} has average epoch loss: {avg_loss}")
   

if __name__ == '__main__':
	main()

	

