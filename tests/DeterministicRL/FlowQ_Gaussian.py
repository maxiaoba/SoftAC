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
from sac.policies import GaussianPolicy, LatentSpacePolicy, GMMPolicy, UniformPolicy
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from examples.variants import parse_domain_and_task, get_variants

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='DRL')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--scale_reward', type=float, default=1)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='FlowQ_Gaussian')
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

states = np.array([[0.5]])

if args.mode == 'train':
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
        'reg': 1e-3,
        'action_prior': 'uniform',
        'reparameterize': True,
        'squash': True,
    }
    value_fn_params = {'layer_size': 256}
    algorithm_params = {    
        'lr': 3e-4,
        'discount': 0.0, # use 0.0 for deterministic transition
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
            'eval_deterministic': False, # True would set policy to be deterministic at evaluation
        }
    }
    replay_buffer_params = {'max_replay_buffer_size': 1e6}
    sampler_params = {
        'max_path_length': 1,
        'min_pool_size': 1000,
        'batch_size': 256,
    }

    from deterministic_rl import DeterministicRLEnv
    env = DeterministicRLEnv(states=states)

    pool = SimpleReplayBuffer(env_spec=env.spec, with_raw_action=True, **replay_buffer_params)

    sampler = SimpleSampler(with_raw_action=True, **sampler_params)pleSampler(**sampler_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    M = value_fn_params['layer_size']
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

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

elif args.mode == 'plot':
    import matplotlib.pyplot as plt
    if not os.path.isdir(log_dir + '/Plots'):
        os.mkdir(log_dir + '/Plots')
    with tf.Session() as sess:
        epoch = args.epoch
        import joblib
        data = joblib.load(log_dir + '/' + 'itr_' + str(epoch) + '.pkl')
        sample_num = 1000

        policy = data['policy']
        from deterministic_rl import DeterministicRLEnv
        env = DeterministicRLEnv(states=states)
        qf1 = data['qf1']
        qf2 = data['qf2']
        vf = data['vf']
        # param1 = policy.get_param_values(trainable=True)
        for state in states:
            plt.figure()
            print('state: ', state)
            actions = policy.get_actions(
                np.repeat(np.array(state)[None, :], sample_num, axis=0))
            ax = plt.subplot(2, 1, 1)
            ax.set_title('a0')
            plt.hist(actions[:, 0], bins=20)
            ax = plt.subplot(2, 1, 2)
            ax.set_title('a1')
            plt.hist(actions[:, 1], bins=20)
            plt.savefig(log_dir + '/Plots/' + 'epoch_' + str(epoch) + 's' + str(state) + '_action_hist.png')
            plt.close()

            plt.figure()
            x = np.linspace(-1, 1, 100)
            y = np.linspace(-1, 1, 100)
            z = np.zeros((100,100))
            for i in range(len(x)):
                for j in range(len(y)):
                    z[j,i] = (env._get_reward([x[i], y[j]]))
            cs = plt.contourf(x, y, z)
            plt.colorbar(cs)
            plt.scatter(actions[:, 0], actions[:, 1])
            plt.savefig(log_dir + '/Plots/' + 'epoch_' + str(epoch) + 's' + str(state) + '_action_scatter.png')


            plt.figure()
            x = np.linspace(-0.99, 0.99, 100)
            y = np.linspace(-0.99, 0.99, 100)
            z1 = np.zeros((100,100))
            z2 = np.zeros((100,100))
            for i in range(len(x)):
                for j in range(len(y)):
                    actions = np.array([[x[i], y[j]]])
                    raw_actions = np.arctanh(actions)
                    z1[j,i] = qf1.eval(np.reshape(state,(1,-1)),actions)[0]
                    z2[j,i] = policy.get_log_pis(np.reshape(state,(1,-1)),actions)[0]
            ax = plt.subplot(2,1,1)
            ax.set_title('qval')
            plt.contourf(x, y, z1)
            ax = plt.subplot(2,1,2)
            ax.set_title('logpi')
            plt.contourf(x, y, z2)
            plt.savefig(log_dir+'/Plots/'+'epoch_'+str(epoch)+'s'+str(state)+'_contour.png')
            plt.close()


