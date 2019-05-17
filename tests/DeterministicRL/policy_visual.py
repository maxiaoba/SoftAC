import os
import gym
import numpy as np
import time
import argparse
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

with tf.Session() as sess:
	sample_num = 1000
	log_dir = './Data/RealNVP'
	itr = 100
	data_path = log_dir+'/itr_'+str(itr)+'.pkl'
	data = joblib.load(data_path)
	env = data['env']
	o = env.reset()
	policy = data['policy']
	max_path_length = 4
	path_length = 0
	done = False
	c_r = 0.0
	if not os.path.isdir(log_dir+'/Plots'):
		os.mkdir(log_dir+'/Plots')
	while (path_length < max_path_length) and (not done):
		plt.figure()
		print('state: ',o)
		path_length += 1
		actions = policy.get_actions(np.repeat(np.array(o)[None,:],sample_num,axis=0))
		plt.scatter(actions[:,0],actions[:,1])
		plt.savefig(log_dir+'/Plots/'+'itr_'+str(itr)+'s'+str(o)+'.png')
		plt.close()
		o, r, done, _ = env.step(actions[0])
		c_r += r
	print('c_r: ',c_r)
