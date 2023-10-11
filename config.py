import jax.numpy as np
import jax

setup = {'algID':           'test',  # Agent Algorithm
         'simInfo':         '_oracle',              # Filename header
         'trial_count':     1,             # number of learning trial (100)
         'episode_count':   2000,          # number of episodes to learn (2000)
         'max_steps':       500,               # max. number of steps in a episode (500)
         'L':               np.array([1.0]),          # probability to give a feedback
         'C':               np.array([0.2]),           # Human feedback confidence level
         'dispON':          False}

parameters = {
    "key":    jax.random.PRNGKey(42),
    "gamma":        0.9,  # discount factor
    "alpha":        0.05, #1.0/128.0 # learning rate
    "eps":          0.1,  # e-greedy policy parametre (probability to select exploration action)
    "type":         2,
    "tempConst":   1.5   # Temperature constant for Bolzmann exploration policy
}

environment = {
    "actions":{'n': np.array([0, 1]), 
               's': np.array([0, -1]), 
               'w': np.array([-1, 0]), 
               'e': np.array([1, 0])},

    "size":{'X':4,'Y':4},

    "obstacles":np.array([  [1,1],
                            [2,1],
                            [3,1],
                            [1,2],
                            [1,3],
                            [2,3],
                            [3,3]]),

    "pellets":np.array([[2,2], [0,0]]),

    "pacman_init_pos":np.array([0,4]),
    "ghost_init_pos":np.array([4,0]),
}

rl = {
    "nStates":(environment['size']['X']*environment['size']['Y'])**2 * len(environment['actions']) * (2**len(environment['pellets'])),
    "nActions":len(environment['actions']),
    "nTrainer":len(setup['C']),
    "stateShape":((environment['size']['X']+1)*(environment['size']['Y']+1), # Position of pacman
                 (environment['size']['X']+1)*(environment['size']['Y']+1), # Position of ghost
                            (2**len(environment['pellets'])),                  # State of pellets (available or consumed)
                            len(environment['actions']))                      # Direction of ghost
    }