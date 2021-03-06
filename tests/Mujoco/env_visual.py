import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='Ant')
parser.add_argument('--ml', type=int, default=1000)
args = parser.parse_args()

from sac.envs import GymEnv
if args.exp_name == 'HumanoidRllab':
	from sac.envs import MultiDirectionHumanoidEnv
	env = MultiDirectionHumanoidEnv()
elif args.exp_name == "SwimmerRllab":
	from sac.envs import MultiDirectionSwimmerEnv
	env = MultiDirectionSwimmerEnv()
else:
	env = GymEnv(args.exp_name+'-v1')
	env.render('human')
from rllab.envs.normalized_env import normalize
env = normalize(env)
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