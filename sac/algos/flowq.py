from numbers import Number

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from .base import RLAlgorithm

from qj import qj

class FlowQ(RLAlgorithm, Serializable):

    def __init__(
            self,
            base_kwargs,

            env,
            policy,
            initial_exploration_policy,
            vf,
            pool,
            plotter=None,

            lr=3e-3,
            clip_gradient=None,
            scale_reward=1,
            min_y=False,
            vf_reg=0.0,
            vf_reg_decay=1.0,
            vf_reg_min=0.0,
            discount=0.99,
            tau=0.01,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,

            save_full_state=False,
    ):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.

            env (`rllab.Env`): rllab environment object.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.

            vf (`ValueFunction`): Soft value function approximator.

            pool (`PoolBase`): Replay buffer to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.

            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.

            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
        """

        Serializable.quick_init(self, locals())
        super(FlowQ, self).__init__(**base_kwargs)

        self._env = env
        self._policy = policy
        self._initial_exploration_policy = initial_exploration_policy
        self._vf = vf
        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._vf_lr = lr
        self._clip_gradient = clip_gradient
        self._scale_reward = scale_reward
        self._min_y = min_y
        self._vf_reg = vf_reg
        self._vf_reg_decay = vf_reg_decay
        self._vf_reg_min = vf_reg_min
        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        # Reparameterize parameter must match between the algorithm and the
        # policy actions are sampled from.
        assert reparameterize == self._policy._reparameterize
        self._reparameterize = reparameterize

        self._save_full_state = save_full_state

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._training_ops = list()

        self._init_placeholders()
        self._init_train_update()
        self._init_target_ops()

        # Initialize all uninitialized variables. This prevents initializing
        # pre-trained policy and qf and vf variables.
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        self._sess.run(tf.variables_initializer(uninit_vars))


    @overrides
    def train(self):
        """Initiate training of the SAC instance."""

        self._train(self._env, self._policy, self._initial_exploration_policy, self._pool)

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_pl = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Da),
            name='actions',
        )
        if self._policy._squash:
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, self._Da),
                name='raw_actions',
            )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

        self._vf_reg_ph = tf.placeholder(
            tf.float32,
            shape=(),
            name='vf_reg',
        )

    @property
    def scale_reward(self):
        if callable(self._scale_reward):
            return self._scale_reward(self._iteration_pl)
        elif isinstance(self._scale_reward, Number):
            return self._scale_reward

        raise ValueError(
            'scale_reward must be either callable or scalar')

    def _init_train_update(self):

        with tf.variable_scope('target'):
            vf_next_target_t = self._vf.get_output_for(self._next_observations_ph)  # N
            self._vf_target_params = self._vf.get_params_internal()

        ys = tf.stop_gradient(
            self.scale_reward * self._rewards_ph +
            (1 - self._terminals_ph) * self._discount * vf_next_target_t
        )  # N

        if self._min_y:
            vf_next_t = self._vf.get_output_for(self._next_observations_ph, reuse=True)
            y2s = tf.stop_gradient(
                self.scale_reward * self._rewards_ph +
                (1 - self._terminals_ph) * self._discount * vf_next_t
            )  # N
            ys = tf.minimum(ys, y2s)

        if self._policy._squash:
            log_pi = self._policy.log_pis_for(self._observations_ph,self._raw_actions_ph)
        else:
            log_pi = self._policy.log_pis_for(self._observations_ph,self._actions_ph)

        self._vf_t = self._vf.get_output_for(self._observations_ph, reuse=True)  # N
        self._vf_params = self._vf.get_params_internal()

        policy_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self._policy.name)
        policy_regularization_loss = tf.reduce_sum(
            policy_regularization_losses)

        q_vpi = self._vf_t + log_pi

        td_loss = tf.reduce_mean(tf.squared_difference(ys,q_vpi))
        self._td_loss = td_loss
        policy_loss = (td_loss
                       + policy_regularization_loss)

        self._vf_reg_loss = tf.reduce_mean(tf.abs(self._vf_t))
        self._vf_loss_t = td_loss + self._vf_reg_ph * self._vf_reg_loss

        p_optimizer = tf.train.AdamOptimizer(self._policy_lr, name="PolicyOptimizer")
        p_gvs = p_optimizer.compute_gradients(policy_loss,var_list=self._policy.get_params_internal())
        p_grads = [x[0] for x in p_gvs]
        p_variables = [x[1] for x in p_gvs]
        if self._clip_gradient is None:
            p_clipped_grads = p_grads
            p_grad_norm = tf.global_norm(p_grads)
        else:
            p_clipped_grads, p_grad_norm = tf.clip_by_global_norm(p_grads, self._clip_gradient)
        policy_train_op = p_optimizer.apply_gradients(zip(p_clipped_grads, p_variables))
        self._p_grad_norm = p_grad_norm

        v_optimizer = tf.train.AdamOptimizer(self._vf_lr, name="VfOptimizer")
        v_gvs = v_optimizer.compute_gradients(self._vf_loss_t,var_list=self._vf_params)
        v_grads = [x[0] for x in v_gvs]
        v_variables = [x[1] for x in v_gvs]
        if self._clip_gradient is None:
            v_clipped_grads = v_grads
            v_grad_norm = tf.global_norm(v_grads)
        else:
            v_clipped_grads, v_grad_norm = tf.clip_by_global_norm(v_grads, self._clip_gradient)
        vf_train_op = v_optimizer.apply_gradients(zip(v_clipped_grads, v_variables))
        self._v_grad_norm = v_grad_norm

        self._training_ops.append(policy_train_op)
        self._training_ops.append(vf_train_op)

    def _init_target_ops(self):
        """Create tensorflow operations for updating target value function."""

        source_params = self._vf_params
        target_params = self._vf_target_params

        self._target_ops = [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target_params, source_params)
        ]

    @overrides
    def _init_training(self, env, policy, pool):
        super(FlowQ, self)._init_training(env, policy, pool)
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        self._vf_reg = np.maximum(self._vf_reg*self._vf_reg_decay,
                                 self._vf_reg_min)

        feed_dict = self._get_feed_dict(iteration, batch)
        self._sess.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        
        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
            self._vf_reg_ph:self._vf_reg,
        }
        if self._policy._squash:
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_pl] = iteration

        return feed_dict

    @overrides
    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        vf, td_loss, vf_reg_loss, p_grad_norm, v_grad_norm = self._sess.run(
            (self._vf_t, self._td_loss, self._vf_reg_loss, self._p_grad_norm, self._v_grad_norm),
            feed_dict)

        logger.record_tabular('vf-avg', np.mean(vf))
        logger.record_tabular('vf-std', np.std(vf))
        logger.record_tabular('vf-grad-norm', v_grad_norm)
        logger.record_tabular('vf-reg-loss', vf_reg_loss)
        logger.record_tabular('vf-reg-weight', self._vf_reg)
        logger.record_tabular('p-grad-norm', p_grad_norm)
        logger.record_tabular('mean-sq-bellman-error1', td_loss)

        self._policy.log_diagnostics(iteration, batch)
        if self._plotter:
            self._plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SAC algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        SAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            snapshot = {
                'epoch': epoch,
                'algo': self
            }
        else:
            snapshot = {
                'epoch': epoch,
                'policy': self._policy,
                'vf': self._vf,
                'env': self._env,
                'vf_reg': self._vf_reg,
            }

        return snapshot

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'vf-params': self._vf.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
            'vf_reg': self._vf_reg,
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)
        self._vf.set_param_values(d['vf-params'])
        self._policy.set_param_values(d['policy-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
        self._vf_reg = d['vf_reg']
