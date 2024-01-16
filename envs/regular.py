import numpy as np
import gym
import matplotlib.pyplot as plt
import os
from collections import defaultdict

class RegularBase:
    def __init__(self) -> None:
        super().__init__()
        self.Q = []
        self.delta = defaultdict(dict)
        self.delta_inv = defaultdict(dict)
        self.sigma = []
        self.q0 = None
        self.q = None
        self.classes = []
        
    def calc_delta_with_table(self, table):
        delta = defaultdict(dict)
        delta_inv = defaultdict(dict)
        for q_cur, a, q_nxt in table:
            delta[q_cur][a] = q_nxt
            delta_inv[q_cur][q_nxt] = a
        return delta, delta_inv
            
    def reset(self):
        self.q = self.q0
        return self.q
    
    def step(self, a):
        assert a in self.sigma
        done, success = False, False
        if self.is_end():
            done, success = True, a == self.q[:-1]
        else:
            assert a in self.delta[self.q].keys()
            self.q = self.delta[self.q][a]
        return self.q, done, success
    
    def sample(self, length):
        raise NotImplementedError
    
    def is_end(self):
        return self.q.endswith('E')

class Parity(RegularBase):
    def __init__(self) -> None:
        super().__init__()
        self.Q = [
            '0', '1', '0E', '1E',
        ]
        self.sigma = [
            '0', '1', 'E'
        ]
        self.classes = ['0', '1']
        self.delta, self.delta_inv = self.calc_delta_with_table([
            ('0', '0', '0'),
            ('0', '1', '1'),
            ('1', '0', '1'),
            ('1', '1', '0'),
            ('0', 'E', '0E'),
            ('1', 'E', '1E'),          
        ])
        self.q0 = '0'
        self.reset()
        
    def sample(self, length):
        assert length > 1
        input = np.random.choice([0, 1], length - 1)
        target = input.sum() % 2
        input_str = ''.join(map(str, input)) + 'E'
        target_str = str(target)
        return (input_str, target_str)

class RegularPOMDP(gym.Env):
    def __init__(
        self, 
        regular_lang,
        length,
        eval_length=None,
        goal_reward=1.0,
        penalty=-1.0,
        add_timestep=False,
    ):
        super().__init__()
        
        self.regular_lang = regular_lang
        self.length = length
        if eval_length is None:
            self.eval_length = length
        else:
            self.eval_length = eval_length
        self.is_eval = False
        
        self.goal_reward = goal_reward
        self.penalty = penalty
        
        self.add_timestep = add_timestep
        
        self.action_space = gym.spaces.Discrete(
            len(self.regular_lang.classes) + 1
        )
        self.action_mapping = {
            i: c for i, c in enumerate(self.regular_lang.classes)
        }
        self.action_mapping[len(self.regular_lang.classes)] = '*'
        
        obs_dim = len(self.regular_lang.sigma)
        if self.add_timestep:
            obs_dim += 1
            
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.prev_q, self.q = None, None
        self.time_step = 0
        self.input, self.target = None, None
        self.current_length = 0
        
    def eval(self):
        self.is_eval = True
        
    def get_obs(self):
        obs = np.zeros(self.observation_space.shape)
        a = self.regular_lang.delta_inv[self.prev_q][self.q]
        obs[self.regular_lang.sigma.index(a)] = 1.0
        if self.add_timestep:
            obs[-1] = self.time_step / self.current_length
        return obs
    
    def reset(self):
        if self.is_eval:
            self.current_length = self.eval_length
        else:
            self.current_length = np.random.randint(2, self.length + 1)
        self.time_step = 0
        self.input, self.target = self.regular_lang.sample(self.current_length)
        self.prev_q = self.regular_lang.reset()
        self.q = self.regular_lang.step(self.input[0])[0]
        return self.get_obs()
    
    def step(self, action):
        self.time_step += 1
        a = self.action_mapping[action]
        if self.time_step == self.current_length:
            if a != '*':
                _, done, success = self.regular_lang.step(a)
                rew = float(success) * self.goal_reward
            else:
                rew, done = self.penalty, True
        else:
            self.prev_q, (self.q, done, success) = self.q, self.regular_lang.step(
                self.input[self.time_step]
            )
            rew = 0.0
                
        return self.get_obs(), rew, done, {}
    
    def get_state(self):
        return [self.prev_q, self.q]
    
class ParityPOMDP(RegularPOMDP):
    def __init__(
        self,
        length,
        eval_length=None,
    ):
        super().__init__(
            Parity(),
            length,
            eval_length=eval_length,
        )
        
if __name__ == '__main__':
    env = ParityPOMDP(10)
    obs = env.reset()
    print(env.input, env.target)
    print(env.time_step, "null", obs)
    done = False
    while not done:
        # if env.time_step < env.current_length - 1:
        #     act = env.action_space.n - 1
        # else:
        act = env.action_space.sample()
        obs, rew, done, _ = env.step(act)
        print(env.time_step, env.action_mapping[act], obs, rew, done)