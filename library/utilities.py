import numpy as onp
import jax
import jax.numpy as np
from itertools import combinations
from functools import partial
from config import parameters, environment, rl, setup

"""
Episodic RL monitor
"""
class RLmon(object):
    
    def __init__(self, numData=1):
        self.episode_count = setup['episode_count']
        self.numData = numData
        # prepare a buffer for averaging reward
        self.ave   = onp.zeros([numData, setup['episode_count']])
        self.aveSq = onp.zeros([numData, setup['episode_count']])
        self.raw   = onp.zeros([numData, setup['episode_count'], setup['trial_count']])
        return
    
    def store(self, episode_idx, trial_idx, d):
        # update averaged reward        
        self.ave[:,episode_idx]   = (self.ave[:,episode_idx]   * trial_idx + d) / (trial_idx + 1)
        self.aveSq[:,episode_idx] = (self.aveSq[:,episode_idx] * trial_idx + d**2) / (trial_idx + 1)
        self.raw[:,episode_idx, trial_idx] = d
        return
    
    def saveData(self, fname):
        stddev = np.sqrt( self.aveSq - self.ave**2 )
        np.savez(fname, ave=self.ave, std=stddev, raw=self.raw)        
        return


# Save results
def save_results(monitors,agent):
    for name,monitor in monitors.items():
        monitor.saveData('results/' + name + '_' + str(setup['algID']) + str(setup['simInfo']))
   
    #fname = 'results/plot_' + str(algID) + str(simInfo)
    #mon.savePlot(fname)

    agent.save('learnedStates/pacman_' + str(setup['algID']))


# ----------------------------------------------------------------------------
# Process data
# ----------------------------------------------------------------------------

def dict_search(my_dict, target_value):
    for key, value in my_dict.items():
        if np.array_equal(value,target_value):
            return key
    return None  # Return None if the value is not found


def moving_average(d, len):
    prePadLen = len//2
    posPadLen = len - prePadLen
    d_ = np.append(d[0:prePadLen], d)
    d_ = np.append(d_, d[-posPadLen:])
    cs = np.cumsum(d_)
    ma = (cs[len:] - cs[:-len]) / len
    return ma

@jax.jit
def mask_array(input_array):
    """
    Generate arrays containing every other element from the input array.

    Args:
    - input_array (array): The input array from which combinations are generated.

    Returns:
    - result_arrays (list of lists): A list of arrays, each of which omits one element from the input array.
    """
    return np.array([list(comb) for comb in combinations(input_array, len(input_array) - 1)])

# @jax.jit
def index_to_element(matrix_shape, index):
    """
    Converts an index for a matrix of any shape and dimensionality
    into the corresponding element in the flattened list.

    Args:
    - matrix_shape (tuple): The shape of the matrix as a tuple (rows, columns, depth, ...).
    - index (tuple): The index for the matrix as a tuple (row, column, depth, ...).

    Returns:
    - element: The corresponding element in the flattened list.
    """
    if len(matrix_shape) != len(index):
        raise ValueError("Matrix shape and index must have the same dimensionality.")

    if not all(0 <= i < dim for i, dim in zip(index, matrix_shape)):
        raise ValueError("Index is out of bounds for the given matrix shape.")

    flattened_index = 0
    multiplier = 1

    for i in range(len(matrix_shape) - 1, -1, -1):
        flattened_index += index[i] * multiplier
        multiplier *= matrix_shape[i]

    return flattened_index


# Define the dictionary mapping input arrays to numbers
array_to_number = {
    (0, 1): 0,    # north
    (0, -1): 1,   # south
    (-1, 0): 2,   # west
    (1, 0): 3     # east
    }

# @jax.jit
def map_array_to_number(input_array):
    # Use the dictionary to map the input array to a number
    return array_to_number.get(tuple(input_array.tolist()))
# ----------------------------------------------------------------------------
# Log calculation functions
# ----------------------------------------------------------------------------

# calculate log(a+b) from log(a) and log(b)
def logadd(a, b):
    if a > b:
        out = a + np.log( 1 + np.exp(b-a) )
    elif a < b:
        out = b + np.log( 1 + np.exp(a-b) )
    else:
        if np.abs(a) == np.inf:
            out = a
        else:
            out = a + np.log( 1 + np.exp(b-a) )
            
    return out
    
# calculate log( sum(a) ) from log(a)
def logsum(a):
    m = np.max(a)
    out = m + np.log( np.sum( np.exp(a-m) ) )
    return out
    
# normalise log-probability p
def lognorm(p):
    m = np.max(p)
    out = p - (m + np.log( np.sum( np.exp(p-m) ) ) )
    return out
        
# Replace character 
def replaceChar(self, st, c, idx):
    return st[0:idx] + c + st[idx+1:]


@profile
def newPos(currPos, dir):
    newPos = currPos + dir
    
    if (all(np.greater_equal(np.array([environment['size']['X'],environment['size']['Y']]),newPos)) and 
        all(np.greater_equal(newPos,np.array([0,0])))) and not np.any(np.all(environment['obstacles'] == newPos, axis=1)):
        return  newPos
    else:
        return currPos


# @jax.jit
def randMove(key):
    return np.array(list(environment['actions'].values())[jax.random.choice(key, np.arange(rl['nActions']))])
    
    
jit_rand = jax.jit(randMove)

# @jax.jit
def st2ob(pacman_pos,ghost_pos,ghost_dir,pellets_valid):

        pPosIdx = (pacman_pos[0]) + (pacman_pos[1])*(environment['size']['X']+1)     # Position of pacman
        gPosIdx = (ghost_pos[0]) + (ghost_pos[1])*(environment['size']['X']+1)       # Position of ghost
        gDirIdx = int(np.where(np.all(np.array(list(environment['actions'].values()))==ghost_dir,axis=1))[0]) #list(environment['actions'].values()).index(ghost_dir.all()) # Direction of ghost
        peltIdx = int(np.sum(np.array([(p+1)*pellets_valid[p] for p in np.arange(len(environment['pellets']))]))) # Status of pellets
        
        return index_to_element(rl['stateShape'],(pPosIdx,gPosIdx,gDirIdx,peltIdx))