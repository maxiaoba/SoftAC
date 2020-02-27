import argparse
import os

import tensorflow as tf
import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator
from rllab import config

from sac.algos.sac2 import SAC

from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp, unflatten
from sac.policies import UniformPolicy
from sac.policies.gaussian_policy_0 import GaussianPolicy
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from examples.variants import parse_domain_and_task, get_variants

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='Hopper')
parser.add_argument('--nmob', type=int, default=0) # nomalize ob
parser.add_argument('--log_dir', type=str, default='SAC2_Gaussian')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--sr', type=float, default=None)
parser.add_argument('--epoch', type=int, default=3000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--args_data', type=str, default=None)
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=500)
args = parser.parse_args()

from rllab.misc import logger
import os.path as osp
pre_dir = './Data/'+args.exp_name
main_dir = args.log_dir\
            +('nmob' if args.nmob==1 else '')\
            +(('lr'+str(args.lr)) if args.lr else '')\
            +(('sr'+str(args.sr)) if args.sr else '')
log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))

seed = args.seed
np.random.seed(seed)
tf.set_random_seed(seed)

tabular_log_file = osp.join(log_dir, 'process.csv')
text_log_file = osp.join(log_dir, 'text.csv')
params_log_file = osp.join(log_dir, 'args.txt')
logger.log_parameters_lite(params_log_file, args)
logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % args.exp_name)

import json
with open(args.exp_name+'_gaussian_variant.json','r') as in_json:
    variants = json.load(in_json)
    variants['seed'] = seed
    variants["algorithm_params"]["base_kwargs"]["n_epochs"] = args.epoch+1

if args.lr:
    variants['algorithm_params']['lr'] = args.lr
if args.sr:
    variants['algorithm_params']["scale_reward"] = args.sr
policy_params = variants['policy_params']
value_fn_params = variants['value_fn_params']
algorithm_params = variants['algorithm_params']
replay_buffer_params = variants['replay_buffer_params']
sampler_params = variants['sampler_params']

with open(osp.join(log_dir,'params.json'),'w') as out_json:
    json.dump(variants,out_json,indent=2)

from sac.envs import GymEnv
if args.exp_name == 'HumanoidRllab':
    from sac.envs import MultiDirectionHumanoidEnv
    env = MultiDirectionHumanoidEnv()
elif args.exp_name == "SwimmerRllab":
    from sac.envs import MultiDirectionSwimmerEnv
    env = MultiDirectionSwimmerEnv()
else:
    env = GymEnv(args.exp_name+'-v1')
from rllab.envs.normalized_env import normalize
env = normalize(env)
# env._wrapped_env.seed(0)

pool = SimpleReplayBuffer(env_spec=env.spec, **replay_buffer_params)

sampler = SimpleSampler(**sampler_params)

base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

M = value_fn_params['layer_size']
qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf')
vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

initial_exploration_policy = UniformPolicy(env_spec=env.spec)


policy = GaussianPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=(M,M),
        reparameterize=policy_params['reparameterize'],
        reg=policy_params['reg'],
        squash=policy_params['squash'],
)

algorithm = SAC(
    base_kwargs=base_kwargs,
    env=env,
    policy=policy,
    initial_exploration_policy=initial_exploration_policy,
    pool=pool,
    qf=qf,
    vf=vf,
    lr=algorithm_params['lr'],
    scale_reward=algorithm_params['scale_reward'],
    discount=algorithm_params['discount'],
    tau=algorithm_params['tau'],
    reparameterize=policy_params['reparameterize'],
    target_update_interval=algorithm_params['target_update_interval'],
    action_prior=policy_params['action_prior'],
    save_full_state=False,
)

algorithm._sess.run(tf.global_variables_initializer())

algorithm.train()
