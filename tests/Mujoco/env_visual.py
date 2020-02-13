import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='Ant')
parser.add_argument('--ml', type=int, default=200)
args = parser.parse_args()

import pybullet_envs
import gym
from sac.envs.rllab_env import RLLabEnv
env = gym.make(args.exp_name+'BulletEnv-v0')
env.seed(0)
env.render('human')
env.reset()

max_path_length = args.ml
path_length = 0
done = False
c_r = 0.0
while (path_length < max_path_length) and (not done):
	path_length += 1
	a = env.action_space.sample()
	o, r, done, _ = env.step(a)
	c_r += r
	# env.render()
	print("step: ",path_length)
	print("o: ",o)
	print("a: ",a)
	print('r: ',r)
	print(done)
	time.sleep(0.1)
print('c_r: ',c_r)
