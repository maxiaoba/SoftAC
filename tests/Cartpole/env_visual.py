import gym
import numpy as np
import time
import argparse

from cartpole import CartPoleEnv 
env = CartPoleEnv()

env.reset()
max_path_length = 200
path_length = 0
done = False
c_r = 0.0
while (path_length < max_path_length) and (not done):
	path_length += 1
	a = env.action_space.sample()
	o, r, done, _ = env.step(a)
	c_r += r
	env.render()
	print("step: ",path_length)
	print("o: ",o)
	print("a: ",a)
	print('r: ',r)
	print(done)
	time.sleep(0.1)
print('c_r: ',c_r)
