import numpy as onp
import jax
import jax.numpy as np

from config import parameters, environment, rl, setup
import library.utilities as ut


# @profile
@jax.jit
def tabQL_ps(Q, p, Ce, prev_obs, prev_act, obs, fb, rw, done):
    """
    Tabular Q-learning with Policy shaping
    """
    # Boltzmann exploration policy
    l_pr = Q[obs,:]/parameters['tempConst']
    
    # policy shaping
    # d = hp - hm # for VI need sum of pos and sum of neg
    
    if parameters['type'] == 1:
        # type1 (general case)
        for trainerIdx in np.arange(rl['nTrainer']):
            for i in range(rl['nActions']):
                l_pr[i] += p[trainerIdx,obs,i] * np.log(Ce[trainerIdx])
    else:
        # type2 (only one optimal action)
        mask = np.array([ut.mask_array(marray) for marray in p[:,obs]])
        l_pr += p[:,obs]*np.log(Ce)+ np.sum(mask*np.log(1-Ce))

    pr = np.exp(l_pr - ut.logsum(l_pr))
    action = np.argmax(pr)
    
    # check if this is the first time...
    if prev_obs is not None:
        # one step TD algorithm
        td_err = rw + parameters['gamma'] * Q[obs, action] * (1-done) - Q[prev_obs, prev_act]
        Q = Q.at[prev_obs, prev_act].set(Q[prev_obs, prev_act] + parameters['alpha'] * td_err)

        # Human feedback updates
        p = p.at[np.arange(rl['nTrainer']), prev_obs, prev_act].set(p[np.arange(rl['nTrainer']), prev_obs, prev_act] + (2*fb - 1))

        # TODO: modify for non-binary case
    # action = np.array(list(environment['actions'].values())[np.argmax(Q[obs,:])])
    pr =np.argmax(Q[obs,:])
    
    # decide action based on pr[] probability distribution
    return pr, Q, p

# @profile
@jax.jit
def Cest(Q,d,hp,hm,Ce):
    """
    consistency level (C) estimation
    The consistency level estimation is based on EM algorithm 
    """
    # Update C estimations    
    # prpare valid (s,a) pairs & s - at least one feedback
    sa_pairs = np.array([(s,a) for s in range(rl['nStates']) for a in range(rl['nActions'])])

    valid = np.ones(rl['nStates'] * rl['nActions']) * False

    for n, (s,a) in enumerate(sa_pairs): 
        if np.sum(d[:,s,a]) > 0:
            valid[n] = True
    valid_sa_pairs = sa_pairs[valid==True]
    valid_s = np.unique(np.array([s for (s,a) in valid_sa_pairs]))
    
    # prepare piror of Optimality flag O (P1/P0) from Q-learning
    # given S,a pairs - if optimal ==1 , else ==0
    ln_P_Q1 = ln_P_Q0  = np.zeros((rl['nStates'], rl['nActions']))

    for s in valid_s:
        # Boltzman exploration policy
        ln_pr = Q[s,:]/parameters['tempConst']
        # normalise
        max_ln_pr = np.max(ln_pr)
        ln_pr = ln_pr - (max_ln_pr + np.log( np.sum( np.exp(ln_pr - max_ln_pr))))
        ln_P_Q1[s,:] = ln_pr
        ln_P_Q0[s,:] = np.log(1.0 - np.exp(ln_pr))
    
    # d = hp - hm
    # Ce = np.ones(nTrainer) * 0.5 # 
    # Ce = Ce # set start point of C (should start from 0.5?)
    ln_P1 = np.zeros((rl['nStates'], rl['nActions']))
    ln_P0 = np.zeros((rl['nStates'], rl['nActions']))

    for k in range(20): # EM iteration
        # E-step (compute posterior of O)
        if parameters['type'] == 1:
            # type1 (general case)
            for (s,a) in valid_sa_pairs:
                ln_P1[s,a] = ln_P_Q1[s,a] + np.sum(d[:,s,a] * np.log(Ce))
                ln_P0[s,a] = ln_P_Q0[s,a] + np.sum(d[:,s,a] * np.log(1.0-Ce))
                ln_P0[s,a] = ln_P0[s,a] - ut.logadd(ln_P0[s,a], ln_P1[s,a])
                ln_P1[s,a] = ln_P1[s,a] - ut.logadd(ln_P0[s,a], ln_P1[s,a])
        else:
            # type2 (only one optimal action)
            for (s,a) in valid_sa_pairs:
                ln_P1[s,a] = ln_P_Q1[s,a] + np.sum(d[:,s,a] * np.log(Ce)) + np.sum(np.sum(d[:,s,np.arange(rl['nActions'])!=a], axis=1) * np.log(1.0-Ce))
                # Equation 2 from advise paper (log)
                ln_P0_ = -np.inf
                for a_ in range(rl['nActions']):
                    if a_ != a:
                        ln_P0_ = ut.logadd(ln_P0_, 
                                            ln_P_Q1[s,a_] +
                                            np.sum(d[:,s,a_] * np.log(Ce)) + 
                                            np.sum(np.sum(d[:,s,np.arange(rl['nActions'])!=a_], axis=1) * np.log(1.0-Ce)))
                ln_P0[s,a] = ln_P0_
                ln_P0[s,a] = ln_P0[s,a] - ut.logadd(ln_P0[s,a], ln_P1[s,a])
                ln_P1[s,a] = ln_P1[s,a] - ut.logadd(ln_P0[s,a], ln_P1[s,a])
        
        # M-step (compute C)
        Ce_old = Ce.copy()
        P1, P0 = np.exp(ln_P1), np.exp(ln_P0)
        for m in range(rl['nTrainer']):
            if np.sum(hp[m,:,:] + hm[m,:,:]) > 0:
                Ce[m] = np.sum(P1 * hp[m,:,:] + P0 * hm[m,:,:]) / np.sum(hp[m,:,:] + hm[m,:,:])
        
        Ce = np.clip(Ce, 0.001, 0.999) 
        
        if np.max(np.abs(Ce - Ce_old)) < 1e-3:
            break; # if Ce does not change much, stop EM iterations

    # set the new Ce (avoid 0.0 and 1.0)
    # agent.Ce = np.clip(Ce, 0.001, 0.999)
    
    return Ce

 # Tabular one step Temporal Difference

# @profile
@jax.jit
def tabQLgreedy(Q, p, Ce, prev_obs, prev_act, obs, fb, rw, done):

    # check if this is the first time...
    # if prev_obs is not None:
        # one step TD algorithm
    td_err = (rw + parameters['gamma'] * np.max(Q[obs])) - Q[prev_obs, prev_act] 
    Q = Q.at[prev_obs, prev_act].set(Q[prev_obs, prev_act] + parameters['alpha'] * td_err)
        
    # Greedy policy (always select the best action)
    # action = np.array(list(environment['actions'].values())[np.argmax(Q[obs,:])])
    pr =np.argmax(Q[obs,:])
    # action = np.array(list(environment['actions'].values())[pr])
    

    return pr, Q, p 

