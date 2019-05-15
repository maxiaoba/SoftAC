import gym
import numpy as np
import time
import argparse
import joblib
import tensorflow as tf

with tf.Session() as sess:
	data_path = './Data/RealNVP/itr_50.pkl'
	data = joblib.load(data_path)
	env = data['env']
	o = env.reset()
	policy = data['policy']
	max_path_length = 200
	path_length = 0
	done = False
	c_r = 0.0
	while (path_length < max_path_length) and (not done):
		path_length += 1
		a, _ = policy.get_action(o)
		o, r, done, _ = env.step(a)
		c_r += r
		env.render()
		print("step: ",path_length)
		print("o: ",o)
		print('r: ',r)
		print(done)
		time.sleep(0.1)
	print('c_r: ',c_r)
