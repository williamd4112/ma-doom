from vizdoom_map.ma_doom_env import DoomSyncMultiPlayerEnvironment
from pygame_rl.scenario.predator_prey_environment import PredatorPreyEnvironment, PredatorPreyEnvironmentOptions
import numpy as np
import random
import time
import logging

from collections import deque
import gym
from gym import error, spaces

import cv2
import sys


class MockGymDoomSyncMultiPlayerEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config, num_players, port):
        self.num_players = num_players
        #self.doom_env = DoomSyncMultiPlayerEnvironment(config, num_players)

        # Get action space from first environment
        # For now, we assume each agent has the same action space
        self.action_space = spaces.Discrete(5)

        # Get observation space from first environment
        # For now, we assume observation shape is fixed as (320, 240)
        self.observation_space = spaces.Box(low=0, high=255, shape=(320, 240, 3))

    def _reset(self):
        # self.doom_env.reset()
        return [np.random.rand(*[320, 240, 3])] * self.num_players

    def _step(self, a):
        info = {}
        #rewards, done = self.doom_env.step(a)
        rewards = [np.random.rand()] * self.num_players
        done = random.choice([True, False])
        #next_states = self.doom_env.current_state()
        next_states = [np.random.rand(*[320, 240, 3])] * self.num_players

        return next_states, rewards, done, info


class GymDoomSyncMultiPlayerEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config, num_players, port=8000):
        self.num_players = num_players
        self.doom_env = DoomSyncMultiPlayerEnvironment(config, num_players, port, deathmatch=False)

        # Get action space from first environment
        # For now, we assume each agent has the same action space
        self.action_space = spaces.Discrete(len(self.doom_env.actions[0]))

        # Get observation space from first environment
        # For now, we assume observation shape is fixed as (320, 240)
        self.observation_space = spaces.Box(low=0, high=255, shape=(320, 240, 3))

    def _reset(self):
        self.doom_env.reset()
        return self.doom_env.current_state()

    def _step(self, a):
        info = {}
        rewards, done = self.doom_env.step(a)
        next_states = self.doom_env.current_state()

        return next_states, rewards, done, info

    def _close(self):
        self.doom_env.close()


class GymPredatorPreySyncMultiPlayerEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_options):
        # Create the environment
        self.env = PredatorPreyEnvironment(env_options=env_options)

        # Get action space
        self.action_space = spaces.Discrete(len(self.env.actions))

        # Render the environment
        self.env.render()

        # Get observation space
        dim = tuple(self.env.renderer.get_screenshot_dim())
        self.observation_space = spaces.Box(low=0, high=255, shape=dim)

    def _reset(self):
        self.env.reset()
        return self.env.state

    def _step(self, a):
        env_obs = self.env.take_action(a)
        reward = env_obs.reward
        observation = env_obs.next_state
        done = observation.is_terminal()
        info = {}
        return observation, reward, done, info


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, res=84, grayscale=True):
        self.grayscale = grayscale
        self.channel = 3 if not grayscale else 1
        gym.ObservationWrapper.__init__(self, env)
        self.res = res
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, self.channel))

    def _observation(self, obs):
        frames = []
        for i, frame in enumerate(obs):
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(frame, (self.res, self.res)))
        #frames = [cv2.resize(frame, (self.res, self.res)) for frame in obs]
        return [frame.reshape((self.res, self.res, self.channel)) for frame in frames]


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, nplayers, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip
        self.nplayers = nplayers

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = np.zeros(self.nplayers)
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class NdarrayEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def _state_to_ndarray(self, states):
        states_expand = [state[np.newaxis, ::] for state in states]
        return np.concatenate(states_expand)

    def _reward_to_ndarray(self, rewards):
        return np.asarray(rewards)

    def _step(self, action):
        next_states, rewards, done, info = self.env.step(action)
        return self._state_to_ndarray(next_states), self._reward_to_ndarray(rewards), done, info

    def _reset(self):
        states = self.env.reset()
        return self._state_to_ndarray(states)


def wrap_ma_doom(config, nplayers, port):
    env = GymDoomSyncMultiPlayerEnvironment(config, nplayers, port)
    env = WarpFrame(env)
    env = NdarrayEnv(env)
    env = MaxAndSkipEnv(env, nplayers)
    return env

def wrap_predator_prey(config):
    object_size = {
        "PREDATOR": config.npredator,
        "PREY": config.nprey,
        "OBSTACLE": config.nobstacle
    }
    print(config.map_path)
    env_options = PredatorPreyEnvironmentOptions(
                        map_path=config.map_path,
                        object_size=object_size,
                        ai_frame_skip=config.frame_skip
                  )
    env = GymPredatorPreySyncMultiPlayerEnvironment(env_options)



def main():
    import argparse
    config = argparse.ArgumentParser()
    config.parse_args()
    config.npredator = 3
    config.nprey = 3
    config.nobstacle = 8
    config.map_path = "data/map/predator_prey/predator_prey_15x15.tmx"
    config.frame_skip = 2
    env = wrap_predator_prey(config)
    env.reset()

    for i in range(10):
        obs, reward, done, info = env.step(0)
        print("{}, {}, {}, {}".format(obs, reward, done, info))



if __name__ == '__main__':
    main()
