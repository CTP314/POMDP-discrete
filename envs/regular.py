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

class Even_Pairs(RegularBase):
# 第一位和最后一位相同
    def __init__(self) -> None:
        super().__init__()
        self.Q = [
            '00', '01', '10', '11', '0E', '1E',
        ]
        self.sigma = [
            '0', '1', 'E'
        ]
        self.classes = ['0', '1']
        self.delta, self.delta_inv = self.calc_delta_with_table([
            ('00', '0', '00'),
            ('00', '1', '01'),
            ('01', '0', '00'),
            ('01', '1', '01'),
            ('10', '0', '10'),
            ('10', '1', '11'),
            ('11', '0', '10'),
            ('11', '1', '11'),
            ('00', 'E', '1E'),
            ('01', 'E', '0E'),
            ('10', 'E', '0E'),
            ('11', 'E', '1E'),     
        ])
        self.q0 = None
        self.reset()
        
    def sample(self, length):
        assert length > 1
        input = np.random.choice([0, 1], length - 1)
        target = 1 if input[0] == input[-1] else 0
        input_str = ''.join(map(str, input)) + 'E'
        target_str = str(target)
        self.q0 = input_str[0] * 2
        return (input_str, target_str)

class Cycle_Navigation(RegularBase):
# 在长为5的环上的位置
    def __init__(self) -> None:
        super().__init__()
        self.Q = [
            '0', '1', '2', '3', '4', '0E', '1E', '2E', '3E', '4E',
        ]
        self.sigma = [
            '0', '1', '2', 'E'
        ]
        self.classes = ['0', '1', '2', '3', '4']
        self.delta, self.delta_inv = self.calc_delta_with_table([
            ('0', '1', '1'),
            ('1', '1', '2'),
            ('2', '1', '3'),
            ('3', '1', '4'),
            ('4', '1', '0'),
            ('0', '2', '4'),
            ('1', '2', '0'),
            ('2', '2', '1'),
            ('3', '2', '2'),
            ('4', '2', '3'),
            ('0', '0', '0'),
            ('1', '0', '1'),
            ('2', '0', '2'),
            ('3', '0', '3'),
            ('4', '0', '4'),
            ('0', 'E', '0E'),
            ('1', 'E', '1E'),
            ('2', 'E', '2E'),
            ('3', 'E', '3E'),
            ('4', 'E', '4E'),     
        ])
        self.q0 = '0'
        self.reset()
        
    def sample(self, length):
        assert length > 1
        input = np.random.choice([0, 1, 2], length - 1)
        target = 0
        for x in input:
            if x == 1: target = (target + 1)%5
            elif x == 2: target = (target - 1)%5
        input_str = ''.join(map(str, input)) + 'E'
        target_str = str(target)
        return (input_str, target_str)

class D2(RegularBase):
# a(ab)*b
    def __init__(self) -> None:
        super().__init__()
        self.Q = [
            'S', '0', '1', '2', 'F', '0E', '1E'
        ]
        self.sigma = [
            '0', '1', 'E'
        ]
        self.classes = ['0', '1']
        self.delta, self.delta_inv = self.calc_delta_with_table([
            ('S', '0', '1'),
            ('S', '1', 'F'),
            ('S', 'E', '0E'),
            ('0', '0', 'F'),
            ('0', '1', '1'),
            ('0', 'E', '0E'),
            ('1', '0', '0'),
            ('1', '1', '2'),
            ('1', 'E', 'F'),
            ('2', '0', 'F'),
            ('2', '1', 'F'),
            ('2', 'E', '1E'),
            ('F', '0', 'F'),
            ('F', '1', 'F'),
            ('F', 'E', '0E')  
        ])
        self.q0 = 'S'
        self.reset()
        
    def sample(self, length):
        assert length > 1
        input = np.random.choice([0, 1], length - 1)
        target = 0
        if input[0] == '0' and input[-1] == '1' and (length - 1)%2 == 0:
            flg = True
            for idx in range(0, (length - 3)//2):
                if input[1 + 2*idx] != '0' or input[2 + 2*idx] != '1': 
                    flg = False
            if flg:
                target = 1
        input_str = ''.join(map(str, input)) + 'E'
        target_str = str(target)
        return (input_str, target_str)

class Tomita_5(RegularBase):
# 0和1的个数都为偶数
    def __init__(self) -> None:
        super().__init__()
        self.Q = [
             '0', '1', '2', '3', '0E', '1E'
        ]
        self.sigma = [
            '0', '1', 'E'
        ]
        self.classes = ['0', '1']
        self.delta, self.delta_inv = self.calc_delta_with_table([
            ('0', '0', '2'),
            ('0', '1', '1'),
            ('0', 'E', '1E'),
            ('1', '0', '3'),
            ('1', '1', '0'),
            ('1', 'E', '0E'),
            ('2', '0', '0'), 
            ('2', '1', '3'),
            ('2', 'E', '0E'),
            ('3', '0', '1'),
            ('3', '1', '2'),
            ('3', 'E', '0E'),
        ])
        self.q0 = '0'
        self.reset()
        
    def sample(self, length):
        assert length > 1
        input = np.random.choice([0, 1], length - 1)
        target = 1 if (input.sum() % 2 == 0 and (length - 1) % 2 == 0) else 0
        input_str = ''.join(map(str, input)) + 'E'
        target_str = str(target)
        return (input_str, target_str)
      
class Tomita_6(RegularBase):
# 0和1的个数的差为3的倍数
    def __init__(self) -> None:
        super().__init__()
        self.Q = [
             '0', '1', '2', '0E', '1E'
        ]
        self.sigma = [
            '0', '1', 'E'
        ]
        self.classes = ['0', '1']
        self.delta, self.delta_inv = self.calc_delta_with_table([
            ('0', '0', '2'),
            ('0', '1', '1'),
            ('0', 'E', '1E'),
            ('1', '0', '0'),
            ('1', '1', '2'),
            ('1', 'E', '0E'),
            ('2', '0', '1'), 
            ('2', '1', '0'),
            ('2', 'E', '0E'),
        ])
        self.q0 = '0'
        self.reset()
        
    def sample(self, length):
        assert length > 1
        input = np.random.choice([0, 1], length - 1)
        target = 1 if (length - 1 - 2 * input.sum()) % 3 == 0 else 0
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
        
class Even_PairsPOMDP(RegularPOMDP):
    def __init__(
        self,
        length,
        eval_length=None,
    ):
        super().__init__(
            Even_Pairs(),
            length,
            eval_length=eval_length,
        )
        
class Cycle_NavigationPOMDP(RegularPOMDP):
    def __init__(
        self,
        length,
        eval_length=None,
    ):
        super().__init__(
            Cycle_Navigation(),
            length,
            eval_length=eval_length,
        )
        
if __name__ == '__main__':
    env = Cycle_NavigationPOMDP(10)
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