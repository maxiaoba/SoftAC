import numpy as np
import time
import argparse
import joblib
import tensorflow as tf
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='Hopper')
parser.add_argument('--log_dir', type=str, default='SAC_Gaussian')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--itr', type=int, default=3000)
args = parser.parse_args()

pre_dir = './Data/'+args.exp_name
main_dir = args.log_dir
log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))

with tf.Session() as sess:
    data_path = osp.join(log_dir,'itr_'+str(args.itr)+'.pkl')
    data = joblib.load(data_path)

    from sac.envs import GymEnv
    if args.exp_name == 'HumanoidRllab':
        from sac.envs import MultiDirectionHumanoidEnv
        env = MultiDirectionHumanoidEnv()
    else:
        env = GymEnv(args.exp_name+'-v1')
        env.render('human')

    o = env.reset()
    policy = data['policy']
    max_path_length = 1000
    path_length = 0
    done = False
    c_r = 0.0

    while  True:
        if done or (path_length >= max_path_length):
            print('c_r: ',c_r)
            import pdb; pdb.set_trace()
            path_length = 0
            o = env.reset()
        path_length += 1
        a, _ = policy.get_action(o)
        o, r, done, _ = env.step(a)
        c_r += r
        env.render()
        print("step: ",path_length)
        print("o: ",o)
        print("a: ",a)
        print('r: ',r)
        print(done)
        time.sleep(0.1)
