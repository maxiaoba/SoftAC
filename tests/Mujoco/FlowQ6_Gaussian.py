import argparse
import os

import tensorflow as tf
import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator
from rllab import config

from sac.algos.flowq6 import FlowQ

from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp, unflatten
from sac.policies import GaussianPolicy, LatentSpacePolicy, GMMPolicy, UniformPolicy
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from examples.variants import parse_domain_and_task, get_variants

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='Hopper')
parser.add_argument('--nmob', type=int, default=0) # nomalize ob in env
parser.add_argument('--log_dir', type=str, default='FlowQ6_Gaussian')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--bs', type=int, default=None)
parser.add_argument('--cg', type=float, default=None)
parser.add_argument('--tui', type=int, default=1) # target update interval
parser.add_argument('--min_y', type=int, default=0)
parser.add_argument('--vf_reg', type=float, default=0.)
parser.add_argument('--vf_reg_decay', type=float, default=1.)
parser.add_argument('--vf_reg_min', type=float, default=0.)
parser.add_argument('--vf_reg_order', type=int, default=1)
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
            +(('bs'+str(args.bs)) if args.bs else '')\
            +(('cg'+str(args.cg)) if args.cg else '')\
            +(('tui'+str(args.tui)) if args.tui>1 else '')\
            +('min_y' if args.min_y==1 else '')\
            +(('vf_reg'+str(args.vf_reg)\
                +(('order'+str(args.vf_reg_order)) 
                 if args.vf_reg_order>1 else '')\
                +(('decay'+str(args.vf_reg_decay)+'min'+str(args.vf_reg_min))\
                 if args.vf_reg_decay<1. else ''))\
             if args.vf_reg>0. else '')
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
    variants["algorithm_params"]["clip_gradient"] = args.cg
    variants["algorithm_params"]['target_update_interval'] = args.tui
    variants["algorithm_params"]["min_y"] = (args.min_y==1)
    variants["algorithm_params"]["vf_reg"] = args.vf_reg
    variants["algorithm_params"]["vf_reg_decay"] = args.vf_reg_decay
    variants["algorithm_params"]["vf_reg_min"] = args.vf_reg_min
    variants["algorithm_params"]["vf_reg_order"] = args.vf_reg_order

if args.lr:
    variants['algorithm_params']['lr'] = args.lr
if args.bs:
    variants['sampler_params']['batch_size'] = args.bs
policy_params = variants['policy_params']
value_fn_params = variants['value_fn_params']
algorithm_params = variants['algorithm_params']
replay_buffer_params = variants['replay_buffer_params']
sampler_params = variants['sampler_params']

with open(osp.join(log_dir,'params.json'),'w') as out_json:
    json.dump(variants,out_json,indent=2)

if args.exp_name == 'HumanoidRllab':
    from sac.envs import MultiDirectionHumanoidEnv
    env = MultiDirectionHumanoidEnv()
else:
    from sac.envs import GymEnv
    env = GymEnv(args.exp_name+'-v1')
from rllab.envs.normalized_env import normalize
env = normalize(env)
# env._wrapped_env.seed(0)

pool = SimpleReplayBuffer(env_spec=env.spec, with_raw_action=True, **replay_buffer_params)

sampler = SimpleSampler(with_raw_action=True, **sampler_params)

base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

M = value_fn_params['layer_size']
vf1 = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='vf1')
vf2 = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='vf2')

initial_exploration_policy = UniformPolicy(env_spec=env.spec)


policy = GaussianPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=(M,M),
        reparameterize=policy_params['reparameterize'],
        reg=policy_params['reg'],
        squash=policy_params['squash'],
)

algorithm = FlowQ(
    base_kwargs=base_kwargs,
    env=env,
    policy=policy,
    initial_exploration_policy=initial_exploration_policy,
    pool=pool,
    vf1=vf1,
    vf2=vf2,
    lr=algorithm_params['lr'],
    clip_gradient=algorithm_params["clip_gradient"],
    scale_reward=algorithm_params['scale_reward'],
    min_y=algorithm_params['min_y'],
    vf_reg=algorithm_params['vf_reg'],
    vf_reg_decay=algorithm_params['vf_reg_decay'],
    vf_reg_min=algorithm_params['vf_reg_min'],
    vf_reg_order=algorithm_params['vf_reg_order'],
    discount=algorithm_params['discount'],
    tau=algorithm_params['tau'],
    reparameterize=policy_params['reparameterize'],
    target_update_interval=algorithm_params['target_update_interval'],
    action_prior=policy_params['action_prior'],
    save_full_state=False,
)

algorithm._sess.run(tf.global_variables_initializer())

algorithm.train()
