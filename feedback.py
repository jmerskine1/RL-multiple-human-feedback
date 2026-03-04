import numpy as np
from scipy.stats import norm

import mylib


# Support functions for calculating probabilities and entropy

def probability_max_gaussian(means, variances, idx, N=20):
    """
    Calculate the probability that one of the Gaussian random variables has the maximum value.
    
    Parameters:
    means (list): List of means of the Gaussian distributions
    variances (list): List of variances of the Gaussian distributions
    idx (int): Index of the Gaussian random variable for which to calculate the probability
    N (int): Number of samples to use for the approximation
    
    Returns:
    float: Probabilities that the specified Gaussian random variable has the maximum value
    """
    p = 1/N * np.arange(N) + 0.5/N
    x0 = norm.ppf(p, loc=means[idx], scale=np.sqrt(variances[idx]))

    probs = np.ones_like(x0)
    for i in range(len(means)):
        if i != idx:
            probs *= norm.cdf((x0 - means[i]) / np.sqrt(variances[i]))
    
    return np.mean(probs)


def belnoulli_entropy(p):
    """
    Calculate the entropy of a Bernoulli distribution.
    
    Parameters:
    p (float): Probability of success (0 <= p <= 1)
    
    Returns:
    float: Entropy of the Bernoulli distribution
    """
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def optimality_entropy(s, a, Q, Nsa, Hp, Hm, psi_for_hr, psi_for_hw, var_Q_base=500.0):
    """
    Calculate the entropy of an optimality flag based on the Q-values and human feedback.
    
    Parameters:
    s (int): State index
    a (int): Action index
    Q (np.ndarray): Q-values for state-action pairs
    Nsa (np.ndarray): Number of visits to state-action pairs
    Hp (np.ndarray): Human positive feedback counts
    Hm (np.ndarray): Human negative feedback counts
    psi_for_hr (float): Weight for positive feedback
    psi_for_hw (float): Weight for negative feedback
    
    Returns:
    float: Entropy of the policy for the given state and action
    """
    # compute log(O(s,a)=1) from human feedback
    ln_pr_o_h = np.sum((Hp[:, s, a]-Hm[:,s,a]) * (psi_for_hr - psi_for_hw))
    # compute the partition function & normalise
    ln_z = -np.inf
    for a_ in range(Q.shape[1]):
        ln_z = mylib.logadd(ln_z, np.sum((Hp[:, s, a_]-Hm[:,s,a_]) * (psi_for_hr - psi_for_hw)))
    ln_pr_o_h -= ln_z
    pr_o_h = np.exp(ln_pr_o_h)
    # pr_o_h = np.clip(np.exp(ln_pr_o_h), 0.001, 0.999)  # clip to avoid numerical issues

    # compute log(O(s,a)=0) from Q-values and Nsa (prob. Q(s,a) > Q(s,a_) for a_ != a)
    mean_Q = Q[s, :]
    var_Q = var_Q_base / (Nsa[s, :] + 0.1)
    pr_o_q = probability_max_gaussian(mean_Q, var_Q, a)
    # pr_o_q = np.clip(pr_o_q, 0.001, 0.999)  # clip to avoid numerical issues

    nActions = Q.shape[1]
    if pr_o_h * pr_o_q == 0:
        pr_o = 0.0
    else:
        pr_o = pr_o_h * pr_o_q * nActions**2 / (pr_o_h * pr_o_q * nActions**2 + (1.0 - pr_o_h) * (1.0 - pr_o_q) / (1-1/nActions)**2)
        
    return belnoulli_entropy(pr_o)

def normalise(x):
    """
    Normalize an array to have zero mean and unit variance.
    
    Parameters:
    x (np.ndarray): Input array to normalize
    
    Returns:
    np.ndarray: Normalized array
    """
    if len(x) == 0 or np.std(x) == 0 or np.std(x) == 0:
        return x
    return (x - np.mean(x)) / np.std(x)


def get_index_for_feedback(U, L):
    U = normalise(U)  # normalise U to have zero mean and unit variance
    if L > 1.0:
        # L is number of feedbacks per episode
        N_fb = min(len(U), int(L))
    else:
        # L is the probability to give a feedback        
        N_fb = len(U) * L
        N_fb = int(N_fb) + 1 if np.random.rand() < (N_fb - int(N_fb)) else int(N_fb) # number of feedbacks
        N_fb = np.maximum(N_fb, 1) # at least one feedback
    if N_fb == len(U):
        # if the number of feedbacks is equal to the length of U, return all indices
        idx = np.arange(len(U))
    else:
        # pick N_fb items with the largest U
        idx = np.argpartition(-U+np.random.randn(len(U)) * 0.1, N_fb)[:N_fb] 
    return idx



# class for storing the trajectory for generating feedback
class Trajectory():
    def __init__(self, state=[], action=[], optimal_action=[], reward=[], done=[]):
        assert len(state) == len(action) == len(optimal_action) == len(reward) == len(done), \
            f"Length of state, action, optimal_action, reward, done must be the same: {len(state)}, {len(action)}, {len(optimal_action)}, {len(reward)}, {len(done)}"
        self.state = state
        self.action = action
        self.optimal_action = optimal_action
        self.reward = reward
        self.done = done
        return

    def reset(self):
        self.state = []
        self.action = []
        self.optimal_action = []
        self.reward = []
        self.done = []
        return

    def append(self, state=None, action=None, optimal_action=None, reward=None, done=None):
        self.state.append(state)
        self.action.append(action)
        self.optimal_action.append(optimal_action)
        self.reward.append(reward)
        self.done.append(done)

    def unique(self):
        # Create a dictionary to track seen (state, action) pairs
        seen = {}
        unique_indices = []
        
        # Find indices of unique (state, action) pairs while preserving order
        for i, (s, a) in enumerate(zip(self.state, self.action)):
            pair = (s, a)
            if pair not in seen:
                seen[pair] = True
                unique_indices.append(i)
        
        # Create new Trajectory with unique entries
        unique_state = [self.state[i] for i in unique_indices]
        unique_action = [self.action[i] for i in unique_indices]
        unique_optimal_action = [self.optimal_action[i] for i in unique_indices]
        unique_reward = [self.reward[i] for i in unique_indices]
        unique_done = [self.done[i] for i in unique_indices]
        
        return Trajectory(
            state=unique_state,
            action=unique_action,
            optimal_action=unique_optimal_action,
            reward=unique_reward,
            done=unique_done
        )

    def overwrite_random_data(self, env_h, oracle_h, trajectroy_len=None):
        # overwrite the trajectory with random state and actions
        if trajectroy_len is None:
            trajectroy_len = len(self.reward)
        
        done = False # done is not used
        self.state = np.random.randint(0, env_h.nStates(), trajectroy_len).tolist()
        self.action = np.random.randint(0, len(env_h.action_list()), trajectroy_len).tolist()
        self.optitmal_action = [np.argmax(oracle_h.Q[s, :]) for s in self.state]
        return
        
    def __str__(self):
        return f"state: {self.state}, action: {self.action}, reward: {self.reward}, done: {self.done}"
    
    def __len__(self):
        return len(self.state)

# class for storing the feedback
class Feedback():
    def __init__(self, state=[], good_actions=[], conf_good_actions=[], bad_actions=[], conf_bad_actions=[]):
        # good_actions = np.array(good_actions)
        # bad_actions = np.array(bad_actions)
        
        # check inputs
        # if len(good_actions.shape) == 1:
        #     good_actions = good_actions.reshape((1,-1))
        # if len(bad_actions.shape) == 1:
        #     bad_actions = bad_actions.reshape((1,-1))
        # if not hasattr(conf_good_actions, '__len__'):
        #     conf_good_actions = [conf_good_actions]
        # if not hasattr(conf_bad_actions, '__len__'):
        #     conf_bad_actions = [conf_bad_actions]
        
        self.state = state
        self.good_actions = good_actions
        self.conf_good_actions = conf_good_actions
        self.bad_actions = bad_actions
        self.conf_bad_actions = conf_bad_actions
        
# generate a single feedback        
def generate_single_feedback(state, action_list, C, optimal_actions, type='binary-feedback', action=None):
    # generate human feedback
    #   action_list = list of possible actions in the environment
    #   C = consistency level (probability of giving a right feedback)
    #   type = feedback type -- 'binary-feedback', 'soft-feedback', 'crisp-set', 'soft-set'
    #   optimal_actions = list of actions for the optimal actions
    #   action = action for giving a feedback
    
    if type=='binary-feedback':
        # right or wrong (original Adivce algorithm)
        if np.random.rand() < C:
            # right feedback
            if action in optimal_actions:
                ret = Feedback(state=state, good_actions=[action], conf_good_actions=1.0)
            else:
                ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
        else:
            # wrong feedback
            if action in optimal_actions:
                ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
            else:
                ret = Feedback(state=state, good_actions=[action], conf_good_actions=1.0)
    elif type=='soft-feedback':
        # right or wrong with the confidence level [0,1]
        sampled_C = np.random.beta(a=C*10, b=(10-C*10)) # sample C from beta distribution to keep the expectation to be the given C
        confidence = np.abs(sampled_C - 0.5) * 2.0
        if np.random.rand() < sampled_C:
            # right feedback
            if action in optimal_actions:
                ret = Feedback(state=state, good_actions=[action], conf_good_actions=confidence)
            else:
                ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=confidence)
        else:
            # wrong feedback
            if action in optimal_actions:
                ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=confidence)
            else:
                ret = Feedback(state=state, good_actions=[action], conf_good_actions=confidence)
    elif type=='crisp-set':
        # set of good and/or bad actions      
        num_feedback_actions = 2
        a_list = np.random.choice(action_list, num_feedback_actions, replace=False)
        good_actions, bad_actions = [], []
        for a in a_list:
            if np.random.rand() < C:
                # right feedback
                if a in optimal_actions:
                    good_actions.append(a)
                else:
                    bad_actions.append(a)
            else:
                # wrong feedback
                if a in optimal_actions:
                    bad_actions.append(a)
                else:
                    good_actions.append(a)
        conf_good_actions = 1.0 if len(good_actions) > 0 else []
        conf_bad_actions = 1.0 if len(bad_actions) > 0 else []
        ret = Feedback(state=state, good_actions=good_actions, conf_good_actions=conf_good_actions,
                                    bad_actions=bad_actions,   conf_bad_actions=conf_bad_actions)
    elif type=='soft-set':
        # set of good and/or bad actions      
        num_feedback_actions = 2
        a_list = np.random.choice(action_list, num_feedback_actions, replace=False)
        good_actions, bad_actions = [], []
        sampled_C = np.random.beta(a=C*10, b=(10-C*10)) # sample C from beta distribution to keep the expectation to be the given C
        confidence = np.abs(sampled_C - 0.5) * 2.0
        for a in a_list:
            if np.random.rand() < sampled_C:
                # right feedback
                if a in optimal_actions:
                    good_actions.append(a)
                else:
                    bad_actions.append(a)
            else:
                # wrong feedback
                if a in optimal_actions:
                    bad_actions.append(a)
                else:
                    good_actions.append(a)
        conf_good_actions = confidence if len(good_actions) > 0 else []
        conf_bad_actions = confidence if len(bad_actions) > 0 else []
        ret = Feedback(state=state, good_actions=good_actions, conf_good_actions=conf_good_actions,
                                    bad_actions=bad_actions,   conf_bad_actions=conf_bad_actions)
    
    elif type=='ranked-feedback':
        # ordinal ranking of actions based on the oracle's value function (with noise C)
        # optimal_actions is expected to be the oracle's Q-values for this state if type is ranked-feedback
        q_values = np.array(optimal_actions)
        if q_values.ndim > 1:
            q_values = q_values.flatten()
            
        # Add noise to Q-values based on C
        noise_scale = (1.0 - C) * (np.max(q_values) - np.min(q_values) + 1e-6)
        noisy_q = q_values + np.random.normal(0, noise_scale, size=q_values.shape)
        
        # Get rankings (higher Q means lower rank number, e.g., rank 0 is best)
        ranks = np.argsort(-noisy_q).tolist()
        
        # Put the bottom two as bad feedback, others as good feedback
        if len(ranks) >= 2:
            good_actions = ranks[:-2]
            bad_actions = ranks[-2:]
        else:
            good_actions = [ranks[0]]
            bad_actions = []
            
        ret = Feedback(state=state, 
                       good_actions=good_actions, 
                       conf_good_actions=C,
                       bad_actions=bad_actions,
                       conf_bad_actions=C)
    
    elif type=='ordinal-feedback':
        # Direct input of action values (e.g. [0.8, 0.7, 0.1, 0.5])
        # optimal_actions is expected to be the oracle's Q-values
        q_values = np.array(optimal_actions)
        if q_values.ndim > 1:
            q_values = q_values.flatten()
            
        # Add noise to Q-values based on C
        noise_scale = (1.0 - C) * (np.max(q_values) - np.min(q_values) + 1e-6)
        noisy_q = q_values + np.random.normal(0, noise_scale, size=q_values.shape)
        
        # Softmax or normalization to get values in [0, 1]
        # Using simple min-max normalization for "ordinal" feel
        v_min, v_max = np.min(noisy_q), np.max(noisy_q)
        if v_max > v_min:
            vals = (noisy_q - v_min) / (v_max - v_min)
        else:
            vals = np.ones_like(noisy_q) / len(noisy_q)
            
        ret = Feedback(state=state, good_actions=vals.tolist(), conf_good_actions=C)
    
    # no information binary-feedbacks
    elif type == 'binary-random':
        # randomly pick right or wrong (original Adivce algorithm)
        if np.random.rand() < 1.0/len(action_list):
            ret = Feedback(state=state, good_actions=[action], conf_good_actions=1.0)
        else:
            ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
    elif type == 'binary-positive':
        # always positive (say right) (original Adivce algorithm)
        ret = Feedback(state=state, good_actions=[action], conf_good_actions=1.0)
        
    elif type == 'binary-negative':         
        # always negative (say wrong) (original Adivce algorithm)
        ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
    else:
        raise ValueError(f"Unknown feedback type: {type}")

    return ret


def get_active_utility(s, a, agent_h, mode='count', no_reward=False, var_Q_base=100):
    """
    Calculate the active learning utility for a single state-action pair.
    
    Parameters:
    s (int): State index
    a (int): Action index
    agent_h (agent): The agent instance
    mode (str): Active learning mode ('count' or 'entropy')
    no_reward (bool): Whether the agent is learning without reward
    var_Q_base (float): Base variance for entropy calculation
    
    Returns:
    float: Utility value (higher means feedback is more needed)
    """
    if mode == 'count':
        if no_reward:
            denom = np.abs(agent_h.hp[:, s, a] - agent_h.hm[:, s, a]).sum()
        else:
            nsa = agent_h.Nsa[s, a] if hasattr(agent_h, 'Nsa') else 0.0
            denom = nsa + np.abs(agent_h.hp[:, s, a] - agent_h.hm[:, s, a]).sum()
        return 1.0 / np.maximum(denom, 0.1)
    
    elif mode == 'entropy':
        if not hasattr(agent_h, 'psi_for_hr'):
            return 1.0 # default high utility if VI parameters are not initialized
        return optimality_entropy(
            s, a, 
            agent_h.Q, 
            agent_h.Nsa if hasattr(agent_h, 'Nsa') else np.zeros_like(agent_h.Q), 
            agent_h.hp, 
            agent_h.hm, 
            agent_h.psi_for_hr, 
            agent_h.psi_for_hw,
            var_Q_base=var_Q_base
        )
    return 0.0

# Generate various type of feedback with a specified Active feedback method
def generate_feedback(trajectory,
                      C,
                      L,
                      action_list, 
                      agent_h, 
                      oracle_h,
                      oracle_h_act, 
                      end_of_episode=False, 
                      no_reward=False,
                      feedback_type = 'binary-feedback', 
                      active_feedback_type=None,
                      var_Q_base=100):
    
    if active_feedback_type is None:
        # feedback at random timing (called at every timestep)
        fb = [[] for n in range(len(C))] # reset feedbacks
        for trainerIdx in np.arange(len(fb)):
            if np.random.rand() < L[trainerIdx]:
                opt_act = oracle_h.Q[trajectory.state[-1], :] if (feedback_type == 'ranked-feedback' or feedback_type == 'ordinal-feedback') else [trajectory.optimal_action[-1]]
                fb[trainerIdx] = [generate_single_feedback(
                                        trajectory.state[-1], 
                                        len(action_list), 
                                        C[trainerIdx], 
                                        opt_act, 
                                        type=feedback_type, 
                                        action=trajectory.action[-1])] # Right feedback
            else:
                fb[trainerIdx] = [Feedback()] # no feedback
    elif 'random' in active_feedback_type:
        # active feedback (random) -- create feedback at the end of the episode
        fb = [[] for n in range(len(C))]
        if end_of_episode:
            U = np.random.rand(len(trajectory)) # random utility values
            for trainerIdx in np.arange(len(fb)):
                idx = get_index_for_feedback(U, L[trainerIdx])
                for n in idx:
                    opt_act = oracle_h.Q[trajectory.state[n], :] if (feedback_type == 'ranked-feedback' or feedback_type == 'ordinal-feedback') else [trajectory.optimal_action[n]]
                    fb[trainerIdx].append(generate_single_feedback(
                                                trajectory.state[n],
                                                len(action_list), 
                                                C[trainerIdx], 
                                                opt_act, 
                                                type=feedback_type, 
                                                action=trajectory.action[n]))
    
    elif 'count' in active_feedback_type:
        # active feedback (weighted-uncertainty-based) -- create feedback at the end of the episode
        fb = [[] for n in range(len(C))] # reset feedbacks
        if end_of_episode:
            # get the number of visitations and feedbacks
            U = np.zeros((len(trajectory),))
            for n, (s, a) in enumerate(zip(trajectory.state, trajectory.action)):
                if no_reward:
                    U[n] = 1.0 / np.maximum(
                                            (
                                            np.abs(agent_h.hp[:, s, a] - agent_h.hm[:, s, a]).sum()
                                            ), 0.1)
                else:
                    U[n] = 1.0 / np.maximum(
                                            (
                                            agent_h.Nsa[s, a] + 
                                            np.abs(agent_h.hp[:, s, a] - agent_h.hm[:, s, a]).sum()
                                            ), 0.1)
            
            for trainerIdx in np.arange(len(fb)):
                idx = get_index_for_feedback(U, L[trainerIdx]) # get the indices for feedback based on U
                for n in idx:
                    opt_act = oracle_h.Q[trajectory.state[n], :] if (feedback_type == 'ranked-feedback' or feedback_type == 'ordinal-feedback') else [trajectory.optimal_action[n]]
                    fb[trainerIdx].append(generate_single_feedback(
                                                trajectory.state[n],
                                                len(action_list), 
                                                C[trainerIdx], 
                                                opt_act, 
                                                type=feedback_type, 
                                                action=trajectory.action[n]))
    elif 'entropy' in active_feedback_type:
        # active feedback (weighted-uncertainty-based) -- create feedback at the end of the episode
        fb = [[] for n in range(len(C))] # reset feedbacks
        if end_of_episode:
            # get the number of visitations and feedbacks
            U = np.zeros((len(trajectory),))
            for n, (s, a) in enumerate(zip(trajectory.state, trajectory.action)):
                U[n] = optimality_entropy(
                                        s, a, 
                                        agent_h.Q, 
                                        agent_h.Nsa, 
                                        agent_h.hp, 
                                        agent_h.hm, 
                                        agent_h.psi_for_hr, 
                                        agent_h.psi_for_hw,
                                        var_Q_base=var_Q_base,)
            
            for trainerIdx in np.arange(len(fb)):
                idx = get_index_for_feedback(U, L[trainerIdx]) # get the indices for feedback based on U
                # U[idx] = np.min(U) # set the minimum value to avoid getting feedback on the same state-action pair
                for n in idx:
                    opt_act = oracle_h.Q[trajectory.state[n], :] if (feedback_type == 'ranked-feedback' or feedback_type == 'ordinal-feedback') else [trajectory.optimal_action[n]]
                    fb[trainerIdx].append(generate_single_feedback(
                                                trajectory.state[n],
                                                len(action_list), 
                                                C[trainerIdx], 
                                                opt_act, 
                                                type=feedback_type, 
                                                action=trajectory.action[n]))
        
    elif 'ideal' in active_feedback_type:
        # active feedback (ideal) -- create feedback at the end of the episode
        fb = [[] for n in range(len(C))]
        if end_of_episode:
            trajectory_len = len(trajectory)

            if 'all_states' in active_feedback_type:
                # append trajectory with random states and actions
                for _ in range(min(0, 500 - trajectory_len)):
                    state = np.random.randint(0, agent_h.nStates())
                    action = np.random.randint(0, len(action_list))
                    # action = np.argmax(agent_h.Q[state, :])
                    optimal_action = np.argmax(oracle_h.Q[state, :])
                    trajectory.append(state=state, action=action, optimal_action=optimal_action, reward=0, done=False)

            regret_on_trajectory = []
            for n, (s, a) in enumerate(zip(trajectory.state, trajectory.action)):
                if 'type2' in active_feedback_type:
                    regret_on_trajectory.append(oracle_h.Q[s, :].max() - oracle_h.Q[s, a]) # type 2
                elif 'type3' in active_feedback_type:
                    regret_on_trajectory.append(np.sqrt(agent_h.Nsa[s, a]) * ( oracle_h_act.Q[s, :].max() - oracle_h_act.Q[s, np.argmax(oracle_h_act.Q[s, :])]) )
                else:
                    regret_on_trajectory.append(oracle_h.Q[s, :].max() - oracle_h.Q[s, np.argmax(agent_h.Q[s, :])]) # type 1
            
            for trainerIdx in np.arange(len(fb)):
                N_fb = trajectory_len * L[trainerIdx]
                N_fb = int(N_fb) + 1 if np.random.rand() < (N_fb - int(N_fb)) else int(N_fb) # number of feedbacks
                if len(regret_on_trajectory) == 1:
                    idx = np.array([0])
                else:
                    idx = np.argpartition(-np.array(regret_on_trajectory) + 
                                            np.random.normal(0, 1.0, len(regret_on_trajectory)),
                                            N_fb)[:N_fb] # pick the item with the smallest regret

                for n in idx:
                    opt_act = oracle_h.Q[trajectory.state[n], :] if (feedback_type == 'ranked-feedback' or feedback_type == 'ordinal-feedback') else [trajectory.optimal_action[n]]
                    fb[trainerIdx].append(generate_single_feedback(
                                                trajectory.state[n],
                                                len(action_list), 
                                                C[trainerIdx], 
                                                opt_act, 
                                                type=feedback_type, 
                                                action=trajectory.action[n]))
    else:
        raise ValueError(f"Unknown active feedback type: {active_feedback_type}")
    return fb


