import numpy as np

from rllab.core.serializable import Serializable

from .replay_buffer import ReplayBuffer


class NormalizeReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, env_spec, max_replay_buffer_size, with_raw_action=False,
                obs_alpha=0.001):
        super(NormalizeReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)

        self._env_spec = env_spec
        self._observation_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        self._with_raw_action = with_raw_action
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(env_spec.observation_space.flat_dim)
        self._obs_var = np.ones(env_spec.observation_space.flat_dim)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        if self._with_raw_action:
            self._raw_actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros(max_replay_buffer_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._top = 0
        self._size = 0

    def _update_obs_estimate(self, obs):
        flat_obs = self._env_spec.observation_space.flatten(obs)
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(flat_obs - self._obs_mean)

    def _normalize_obs(self, obs):
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        if self._with_raw_action:
            self._raw_actions[self._top] = kwargs['raw_action']

        self._advance()
        self._update_obs_estimate(observation)
        self._update_obs_estimate(next_observation)

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch =  dict(
            observations=self._normalize_obs(self._observations[indices]),
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._normalize_obs(self._next_obs[indices]),
        )
        if self._with_raw_action:
            batch['raw_actions'] = self._raw_actions[indices]
        return batch

    @property
    def size(self):
        return self._size

    def __getstate__(self):
        d = super(SimpleReplayBuffer, self).__getstate__()
        if self._with_raw_action:
            d.update(dict(
                o=self._observations.tobytes(),
                a=self._actions.tobytes(),
                ra=self._raw_actions.tobytes(),
                r=self._rewards.tobytes(),
                t=self._terminals.tobytes(),
                no=self._next_obs.tobytes(),
                top=self._top,
                size=self._size,
            ))
        else:
            d.update(dict(
                o=self._observations.tobytes(),
                a=self._actions.tobytes(),
                r=self._rewards.tobytes(),
                t=self._terminals.tobytes(),
                no=self._next_obs.tobytes(),
                top=self._top,
                size=self._size,
            ))
        return d

    def __setstate__(self, d):
        super(SimpleReplayBuffer, self).__setstate__(d)
        self._observations = np.fromstring(d['o']).reshape(
            self._max_buffer_size, -1
        )
        self._next_obs = np.fromstring(d['no']).reshape(
            self._max_buffer_size, -1
        )
        self._actions = np.fromstring(d['a']).reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_buffer_size)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8)
        self._top = d['top']
        self._size = d['size']

    def rollout(self, env, policy, path_length, render=False, speedup=None):
        Da = env.action_space.flat_dim
        Do = env.observation_space.flat_dim

        observation = env.reset()
        policy.reset()

        observations = np.zeros((path_length + 1, Do))
        actions = np.zeros((path_length, Da))
        terminals = np.zeros((path_length, ))
        rewards = np.zeros((path_length, ))
        agent_infos = []
        env_infos = []

        t = 0
        for t in range(path_length):

            action, agent_info = policy.get_action(self._normalize_obs(observation))
            next_obs, reward, terminal, env_info = env.step(action)

            agent_infos.append(agent_info)
            env_infos.append(env_info)

            actions[t] = action
            terminals[t] = terminal
            rewards[t] = reward
            observations[t] = observation

            observation = next_obs

            if render:
                env.render()
                time_step = 0.05
                time.sleep(time_step / speedup)

            if terminal:
                break

        observations[t + 1] = observation

        path = {
            'observations': observations[:t + 1],
            'actions': actions[:t + 1],
            'rewards': rewards[:t + 1],
            'terminals': terminals[:t + 1],
            'next_observations': observations[1:t + 2],
            'agent_infos': agent_infos,
            'env_infos': env_infos
        }

        return path


    def rollouts(self, env, policy, path_length, n_paths):
        paths = [
            self.rollout(env, policy, path_length)
            for i in range(n_paths)
        ]

        return paths

