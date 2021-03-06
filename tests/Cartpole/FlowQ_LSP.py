import argparse
import os

import tensorflow as tf
import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator
from rllab import config

from sac.algos.flowq import FlowQ

from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp, unflatten
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from examples.variants import parse_domain_and_task, get_variants

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='cartpole')
parser.add_argument('--scale_reward', type=float, default=1)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='FlowQ_LSP')
parser.add_argument('--args_data', type=str, default=None)
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
args = parser.parse_args()

from rllab.misc import logger
import os.path as osp
pre_dir = './Data/'+args.exp_name
main_dir = args.log_dir+'rs'+str(args.scale_reward)
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

policy_params = {
    'preprocessing_hidden_sizes': (256, 256, 4),
    's_t_units': 2, # num of units of the realNVP inner mlp
    'coupling_layers': 2,
    's_t_layers': 1, # num of layers of the realNVP inner mlp
    'action_prior': 'uniform', # this is a prior for action distribution, not latent distribution
    'preprocessing_output_nonlinearity': 'relu',
    'reparameterize': True,
    'squash': True, # Ture to add tanh on the output
}
value_fn_params = {'layer_size': 256}
algorithm_params = {    
    'lr': 3e-4,
    'discount': 0.99,
    'target_update_interval': 1,
    'tau': 0.005,
    'reparameterize': True,
    'scale_reward': args.scale_reward,
    'base_kwargs': {
        'n_epochs': args.epoch+1,
        'epoch_length': 1000, # number of sample() and training done in one epoch
        'n_train_repeat': 1,
        'n_initial_exploration_steps': 1000,
        'eval_render': False,
        'eval_n_episodes': 10,
        'eval_deterministic': True,
    }
}
replay_buffer_params = {'max_replay_buffer_size': 1e6}
sampler_params = {
    'max_path_length': 100,
    'min_pool_size': 1000,
    'batch_size': 256,
}

from cartpole import CartPoleEnv
env = CartPoleEnv()

pool = SimpleReplayBuffer(env_spec=env.spec, with_raw_action=True, **replay_buffer_params)

sampler = SimpleSampler(with_raw_action=True, **sampler_params)

base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

M = value_fn_params['layer_size']
qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

from sac.policies.uniform_policy import UniformPolicy
initial_exploration_policy = UniformPolicy(env_spec=env.spec)


nonlinearity = {
    None: None,
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh
}[policy_params['preprocessing_output_nonlinearity']]

preprocessing_hidden_sizes = policy_params.get('preprocessing_hidden_sizes')
if preprocessing_hidden_sizes is not None:
    observations_preprocessor = MLPPreprocessor(
        env_spec=env.spec,
        layer_sizes=preprocessing_hidden_sizes,
        output_nonlinearity=nonlinearity)
else:
    observations_preprocessor = None

policy_s_t_layers = policy_params['s_t_layers']
policy_s_t_units = policy_params['s_t_units']
s_t_hidden_sizes = [policy_s_t_units] * policy_s_t_layers

bijector_config = {
    'num_coupling_layers': policy_params['coupling_layers'],
    'translation_hidden_sizes': s_t_hidden_sizes,
    'scale_hidden_sizes': s_t_hidden_sizes,
}

from sac.policies.latent_space_policy import LatentSpacePolicy
policy = LatentSpacePolicy(
    env_spec=env.spec,
    squash=policy_params['squash'],
    bijector_config=bijector_config,
    reparameterize=policy_params['reparameterize'],
    q_function=qf1,
    observations_preprocessor=observations_preprocessor)

algorithm = FlowQ(
    base_kwargs=base_kwargs,
    env=env,
    policy=policy,
    initial_exploration_policy=initial_exploration_policy,
    pool=pool,
    qf1=qf1,
    qf2=qf2,
    vf=vf,
    lr=algorithm_params['lr'],
    scale_reward=algorithm_params['scale_reward'],
    discount=algorithm_params['discount'],
    tau=algorithm_params['tau'],
    reparameterize=algorithm_params['reparameterize'],
    target_update_interval=algorithm_params['target_update_interval'],
    action_prior=policy_params['action_prior'],
    save_full_state=False,
)

algorithm._sess.run(tf.global_variables_initializer())

algorithm.train()
