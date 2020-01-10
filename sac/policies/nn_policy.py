import tensorflow as tf

from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
from sandbox.rocky.tf.policies.base import Policy


class NNPolicy(Policy, Serializable):
    def __init__(self, env_spec, observation_ph, actions,
                 scope_name=None):
        Serializable.quick_init(self, locals())

        self._observations_ph = observation_ph
        self._actions = actions
        self._scope_name = (
            tf.get_variable_scope().name if not scope_name else scope_name
        )
        super(NNPolicy, self).__init__(env_spec)

    @overrides
    def get_action(self, observation, with_raw_action=False):
        """Sample single action based on the observations."""
        if with_raw_action:
            actions, raw_actions = self.get_actions(observation[None],True)
            return actions[0], raw_actions[0], {}
        else:
            return self.get_actions(observation[None])[0], {}

    @overrides
    def get_actions(self, observations, with_raw_actions=False):
        """Sample actions based on the observations."""
        feed_dict = {self._observations_ph: observations}
        if with_raw_actions:
            actions, raw_actions = \
                tf.get_default_session().run([self._actions,self._raw_actions],
                                         feed_dict)
            return actions, raw_actions
        else:
            actions = tf.get_default_session().run(self._actions, feed_dict)
            return actions

    def get_log_pis(self, observations, actions):
        feed_dict = {self._observations_ph: observations, self._actions_ph: actions}
        log_pis = tf.get_default_session().run(self._log_pis, feed_dict)
        return log_pis

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        if tags:
            raise NotImplementedError
        scope = self._scope_name
        # Add "/" to 'scope' unless it's empty (otherwise get_collection will
        # return all parameters that start with 'scope'.
        scope = scope if scope == '' else scope + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
