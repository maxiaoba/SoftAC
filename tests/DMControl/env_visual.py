import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, default='fish')
parser.add_argument('--task', type=str, default='swim')
parser.add_argument('--ml', type=int, default=200)
args = parser.parse_args()

if args.domain == 'show':
	from dm_control import suite
	from dm2gym import make
	for domain_name, task_name in suite.BENCHMARKING:
		env = make(domain_name=domain_name, task_name=task_name)
		print(domain_name, task_name, env.observation_space, env.action_space)
	assert False
from dm2gym import make
env = make(domain_name=args.domain, task_name=args.task)
from rllab.envs.normalized_env import normalize
env = normalize(env)
# env.render('human')
o = env.reset()
# print(env.observation_space.high)
max_path_length = args.ml
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
	print("o_max: ",np.max(o),np.argmax(o))
	print("o_mean: ",np.mean(o))
	print("a: ",a)
	print('r: ',r)
	print(done)
	time.sleep(0.1)
print('c_r: ',c_r)