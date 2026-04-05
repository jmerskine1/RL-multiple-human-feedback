# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:43:24 2018

@author: taku.yamagata

Agent class
"""
import numpy as np
from scipy.special import psi
import pickle
import random

import mylib

class ReplayBuffer():
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

class agent():

    def __init__(self, algID, nStates, nActions, a=100.0, b=100.0, C_fixed=None):
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.reset(algID, nStates, nActions, a, b, C_fixed)
        return
        
        
    def reset(self, algID, nStates, nActions, a, b, C_fixed):
        self.gamma    = 0.9 # discount factor
        self.alpha    = 0.05 #1.0/128.0 # learning rate
        self.eps      = 0.1 # e-greedy policy parametre (probability to select exploration action)
        self.batch_size = 32
        
        self.Q        = None         # value function - Q
        self.prev_obs = None         # previous observation
        self.prev_act = None         # previous action
        
        self.d        = []           # delta for human feedback
        self.Cb       = 0.75         # Confidence level of human feedback
        self.mu       = []
        
        self.hp = []
        self.hm = []
        self.Ce = []
        self.numFeedBacks = []
        self.absQ = 1
        
        self.tempConst = 1.5         # Temperature constant for Bolzmann exploration policy
        
        self.algID    = algID
        self.nStates  = nStates
        self.nActions = nActions
        
        # Variational Inference parameters
        self.a = a # prior parameter for C
        self.b = b # prior parameter for C
        # Fixed C (without C estimation)
        self.C_fixed = C_fixed
        # clear replay buffer
        self.replay_buffer.clear()
        return
    
    
    def act(self, a, obs, rw, done, fb, C, update_Cest=None):

        action = 0 # default action.
        
        if update_Cest is None:
            update_Cest = done

        # state-action wise Expectation Maximisation based C estimation agent
        if self.algID == 'tabQL_Cest_em_org_t1':
            action = self.tabQL_Cest_org(obs, rw, done, fb, type=1)
        elif self.algID == 'tabQL_Cest_em_org_t2':
            action = self.tabQL_Cest_org(obs, rw, done, fb, type=2)
            
        # Expectation Maximisation based C estimation agent
        elif self.algID == 'tabQL_Cest_em_t1':
            action = self.tabQL_Cest_em(obs, rw, done, fb, type=1, update_Cest=update_Cest)
        elif self.algID == 'tabQL_Cest_em_t2':
            action = self.tabQL_Cest_em(obs, rw, done, fb, type=2, update_Cest=update_Cest)
        
        # Variational Inference based C estimation agent            
        elif self.algID == 'tabQL_Cest_vi_t1':
            action = self.tabQL_Cest_vi(obs, rw, done, fb, type=1, update_Cest=update_Cest)
        elif self.algID == 'tabQL_Cest_vi_t2':
            action = self.tabQL_Cest_vi(obs, rw, done, fb, type=2, update_Cest=update_Cest)
        
        # Q-learning (no feedback)
        elif self.algID == 'tabQLgreedy':
            action = self.tabQLgreedy(obs, rw)

        # Q-learning (no feedback)
        elif self.algID == 'tabQLrandom':
            action = self.tabQLrandom(obs, rw)
        # Add these to your existing elif chain in the act() method:

        elif self.algID == 'tabQL_IPL':
            action = self.tabQL_IPL(obs, rw, done, fb, type=1, update_Cest=update_Cest)

        elif self.algID == 'tabQL_visit_active':
            action = self.tabQL_visit_active(obs, rw, done, fb, type=1, update_Cest=update_Cest)

        elif self.algID == 'tabQL_majority_vote':
            action = self.tabQL_majority_vote(obs, rw, done, fb, type=1, update_Cest=update_Cest)

        elif self.algID == 'tabQL_uncertainty_active':
            action = self.tabQL_uncertainty_active(obs, rw, done, fb, type=1, update_Cest=update_Cest)

        elif self.algID == 'tabQL_dawid_skene':
            action = self.tabQL_dawid_skene(obs, rw, done, fb, type=1, update_Cest=update_Cest)

        else:
            raise ValueError(f'{self.algID}: Invalid algorithm option ID')
        
        # store the current observation/action for the next call
        self.prev_obs = obs
        self.prev_act = action

        return action
    
    def update_from_demo(self, state, action):
            # initialise Q
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions]) # initialise Q function (s,a)
        self.Q[state][action] += 1  # Very simple imitation (can be turned into a softmax or ML update)
    # @profile
    def _collect_feedback(self, feedback_list):
        # collect feedback and update self.hp and self.hm. 
        for n, fb_list in enumerate(feedback_list): # n for trainer index
            for fb in fb_list:
                if not hasattr(fb, 'state') or fb.state is None:
                    continue
                
                # Check for ranked or ordinal feedback
                good_actions = fb.good_actions if hasattr(fb, 'good_actions') else []
                bad_actions = fb.bad_actions if hasattr(fb, 'bad_actions') else []
                
                # Check for ordinal feedback (list of floats)
                if len(good_actions) > 0 and isinstance(good_actions[0], float):
                    vals = good_actions
                    for i, v in enumerate(vals):
                        # Distance from 0.5 determines confidence (near 0.5 = low confidence)
                        confidence = np.abs(v - 0.5) * 2.0
                        if v > 0.5:
                            # Better than 0.5: positive advice
                            self.hp[n, fb.state, i] += confidence * fb.conf_good_actions
                        elif v < 0.5:
                            # Worse than 0.5: negative advice
                            self.hm[n, fb.state, i] += confidence * fb.conf_good_actions

                # ranked-feedback now provides good_actions and bad_actions
                # with bottom two in bad_actions.
                elif hasattr(fb, 'good_actions') and hasattr(fb, 'bad_actions') and (len(fb.good_actions) + len(fb.bad_actions) > 2):
                    good_actions = fb.good_actions
                    bad_actions = fb.bad_actions
                    
                    # Update good actions with confidence based on rank
                    for i, a in enumerate(good_actions):
                        # i=0 is best. Confidence decreases as rank increases.
                        conf = fb.conf_good_actions * (1.0 - i / (len(good_actions) + len(bad_actions) - 1))
                        self.hp[n, fb.state, int(a)] += conf
                    
                    # Update bad actions
                    for i, a in enumerate(bad_actions):
                        # Last index is worst.
                        total_idx = len(good_actions) + i
                        conf = fb.conf_bad_actions * (total_idx / (len(good_actions) + len(bad_actions) - 1))
                        self.hm[n, fb.state, int(a)] += conf

                else:
                    # Traditional good/bad action sets (or single-action feedback)
                    if hasattr(fb, 'good_actions'):
                        for a in fb.good_actions:
                            self.hp[n, fb.state, int(a)] = self.hp[n, fb.state, int(a)] + fb.conf_good_actions
                    if hasattr(fb, 'bad_actions'):
                        for a in fb.bad_actions:
                            self.hm[n, fb.state, int(a)] = self.hm[n, fb.state, int(a)] + fb.conf_bad_actions
                            
    # Tabular one step Temporal Difference
    def tabQLgreedy(self, obs, rw):
       
        # initialise Q
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions]) # initialise Q function (s,a)
            
        curr_state_idx = obs
        
        # check if this is the first time...
        if self.prev_obs is not None:
            # one step TD algorithm
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            td_err = (rw + self.gamma * max(self.Q[curr_state_idx,:])) - self.Q[prev_state_idx, prev_action_idx] 
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
        # print("Q:", self.Q[curr_state_idx])
        max_value = np.max(self.Q[curr_state_idx])  # Maximum value for the current state 
        indices = np.flatnonzero(self.Q[curr_state_idx] == max_value)  # Indices of all max values
        # print("indices: ",indices)
        action = np.random.choice(indices)  # Randomly pick one of them

        # Greedy policy (always select the best action)
        # action = np.argmax(self.Q[curr_state_idx,:])

        return action
    
    # Tabular one step Temporal Difference
    def tabQLrandom(self, obs, rw):
       
        # initialise Q
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions]) # initialise Q function (s,a)
            
        curr_state_idx = obs
        
        # check if this is the first time...
        if self.prev_obs is not None:
            # one step TD algorithm
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
        
        # print("Q:", self.Q[curr_state_idx])
        
        
        # print("indices: ",indices)
        
        action = np.random.randint(len(self.Q[curr_state_idx]))
        # Greedy policy (always select the best action)
        # action = np.argmax(self.Q[curr_state_idx,:])

        return action

    def tabQL_Cest_vi(self, obs, rw, done, fb, type=1, update_Cest=False):
        """
        Tabular Q-learning with Policy shaping & consistency level (C) estimation
        The consistency level estimation is based on Variational Inference (VI) algorithm 
        """
        assert self.C_fixed == None, "C_fixed must be None for tabQL_Cest_vi"
    
        # initialise Q
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions])    # initialise Q function (s,a)
            
            self.nTrainer = len(fb)
            self.hp = np.ones([self.nTrainer, self.nStates, self.nActions]) * 0.0
            self.hm = np.ones([self.nTrainer, self.nStates, self.nActions]) * 0.0
            self.Ce = np.ones(self.nTrainer) * 0.5
            self.sum_of_right_feedback = np.zeros(self.nTrainer) # sum of the expected number of right feedbacks (\sum h^r)
            self.sum_of_wrong_feedback = np.zeros(self.nTrainer) # sum of the expected number of wrong feedbacks (\sum h^w)
            self.psi_for_hr = psi(self.sum_of_right_feedback + self.a) - psi(self.sum_of_right_feedback + self.sum_of_wrong_feedback + self.a + self.b) # psi( \sum h^r + alpha ) - psi( \sum h^r + \sum h^w + alpha + beta )
            self.psi_for_hw = psi(self.sum_of_wrong_feedback + self.b) - psi(self.sum_of_right_feedback + self.sum_of_wrong_feedback + self.a + self.b) # psi( \sum h^w + alpha ) - psi( \sum h^r + \sum h^w + alpha + beta )
            self.Nsa = np.zeros((self.nStates, self.nActions)) # number of the agent visiting (s,a) pair
            
            # set prior parameters for C
            if hasattr(self.a, "__len__"):
                self.a = np.array(self.a)
            else:
                self.a = np.ones(self.nTrainer) * self.a
            if hasattr(self.b, "__len__"):
                self.b = np.array(self.b)
            else:
                self.b = np.ones(self.nTrainer) * self.b
                       
        curr_state_idx = obs

        # Boltzmann exploration policy
        l_pr = self.Q[curr_state_idx,:]/self.tempConst
        
        # policy shaping
        if type == 1:
            # type1 (general case)
            for trainerIdx in range(self.nTrainer):
                for i in range(self.nActions):
                    l_pr[i] += self.hp[trainerIdx, curr_state_idx, i] * self.psi_for_hr[trainerIdx] + \
                               self.hm[trainerIdx, curr_state_idx, i] * self.psi_for_hw[trainerIdx]
        else:
            # type2 (only one optimal action)
            for trainerIdx in range(self.nTrainer):
                for i in range(self.nActions):
                    l_pr[i] += np.sum(self.hp[trainerIdx, curr_state_idx, i] * self.psi_for_hr[trainerIdx] + 
                                      self.hm[trainerIdx, curr_state_idx, i] * self.psi_for_hw[trainerIdx]) \
                             - np.sum(self.hm[trainerIdx, curr_state_idx, i] * self.psi_for_hr[trainerIdx] + 
                                      self.hp[trainerIdx, curr_state_idx, i] * self.psi_for_hw[trainerIdx])
            
        l_pr_norm = l_pr - mylib.logsum(l_pr)            
        pr = np.exp(l_pr_norm)
        
        # decide action based on pr[] probability distribution
        action = np.min( np.where( np.random.rand() < np.cumsum(pr) ) )        
        
        # update Nsa (number of the agent visiting (s,a) pair)
        self.Nsa[curr_state_idx, action] += 1

        # check if this is the first time...
        if self.prev_obs is not None:
            # push trajectory to replay buffer
            self.replay_buffer.push(self.prev_obs, self.prev_act, rw, curr_state_idx, done)

            # one step TD algorithm
            
            if len(self.replay_buffer) >= self.batch_size:
                trajectories = zip(*self.replay_buffer.sample(self.batch_size))
                for state_, action_, reward_, next_state_, done_ in zip(*trajectories):
                    # If this is not the first step, apply Q-learning update
                    self.Q[state_, action_] += \
                        self.alpha * (reward_ + (done_ == False) * self.gamma * np.max(self.Q[next_state_]) - self.Q[state_, action_])
            
            # prev_state_idx = self.prev_obs
            # prev_action_idx = self.prev_act
            # max_a_idx = np.argmax(pr)
            # if done:
            #     td_err = (rw) - self.Q[prev_state_idx, prev_action_idx]             
            # else:
            #     td_err = (rw + self.gamma * self.Q[curr_state_idx, max_a_idx]) - self.Q[prev_state_idx, prev_action_idx] 
            # self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
        
            # Human feedback updates
            self._collect_feedback(fb)
        
        # Update C estimations    
        if update_Cest:
            # prpare valid (s,a) pairs & s - at least one feedback
            sa_pairs = np.array([(s,a) for s in range(self.nStates) for a in range(self.nActions)])
            valid = np.ones(self.nStates * self.nActions) * False
            for n, (s,a) in enumerate(sa_pairs): 
                if np.sum(self.hp[:,s,a] + self.hm[:,s,a]) > 0:
                    valid[n] = True
            valid_sa_pairs = sa_pairs[valid==True]
            valid_s = np.unique([s for (s,a) in valid_sa_pairs])
            
            # prepare piror of Optimality flag O (P1/P0) from Q-learning
            ln_P_Q1 = np.zeros((self.nStates, self.nActions))
            ln_P_Q0 = np.zeros((self.nStates, self.nActions))
            for s in valid_s:
                # Boltzman exploration policy
                ln_pr = self.Q[s,:]/self.tempConst
                # normalise
                max_ln_pr = np.max(ln_pr)
                ln_pr = ln_pr - (max_ln_pr + np.log( np.sum( np.exp(ln_pr - max_ln_pr))))
                ln_P_Q1[s,:] = ln_pr
                ln_P_Q0[s,:] = np.log(1.0 - np.exp(ln_P_Q1[s,:]))
            
            Ce = self.Ce
            ln_P1 = np.zeros((self.nStates, self.nActions))
            ln_P0 = np.zeros((self.nStates, self.nActions))
            for k in range(20): # EM iteration
                # compute posterior of O
                if type == 1:
                    # type1 (general case)
                    for (s,a) in valid_sa_pairs:
                        ln_P1[s,a] = np.sum(self.hp[:,s,a] * self.psi_for_hr + self.hm[:,s,a] * self.psi_for_hw) + ln_P_Q1[s,a]
                        ln_P0[s,a] = np.sum(self.hm[:,s,a] * self.psi_for_hr + self.hp[:,s,a] * self.psi_for_hw) + ln_P_Q0[s,a]
                        ln_partition = mylib.logadd(ln_P0[s,a], ln_P1[s,a])
                        ln_P0[s,a] = ln_P0[s,a] - ln_partition
                        ln_P1[s,a] = ln_P1[s,a] - ln_partition
                else:
                    # type2 (only one optimal action)
                    for (s,a) in valid_sa_pairs:
                        ln_P1_ = ln_P_Q1[s,:].copy()
                        for a_ in range(self.nActions):
                            ln_P1_[a_] += np.sum(self.hp[:,s,a_] * self.psi_for_hr + self.hm[:,s,a_] * self.psi_for_hw) - np.sum(self.hm[:,s,a_] * self.psi_for_hr + self.hp[:,s,a_] * self.psi_for_hw)    
                        ln_P1_ = ln_P1_ - mylib.logsum(ln_P1_)
                        ln_P1[s,a] = ln_P1_[a]
                        if np.exp(ln_P1[s,a]) == 1.0:
                            ln_P0[s,a] = - np.inf
                        else:
                            ln_P0[s,a] = np.log(1.0 - np.exp(ln_P1[s,a]))

                # compute posterior of C)
                Ce_old = Ce.copy()
                P1, P0 = np.exp(ln_P1), np.exp(ln_P0)
                for m in range(self.nTrainer):
                    if np.sum(self.hp[m,:,:] + self.hm[m,:,:]) > 0:
                        self.sum_of_right_feedback[m] = np.sum(P1*self.hp[m,:,:] + P0*self.hm[m,:,:])
                        self.sum_of_wrong_feedback[m] = np.sum(P0*self.hp[m,:,:] + P1*self.hm[m,:,:])
                        self.psi_for_hr[m] = psi(self.sum_of_right_feedback[m] + self.a[m]) - psi(self.sum_of_right_feedback[m] + self.sum_of_wrong_feedback[m] + self.a[m] + self.b[m]) # psi( \sum h^r + alpha ) - psi( \sum h^r + \sum h^w + alpha + beta )
                        self.psi_for_hw[m] = psi(self.sum_of_wrong_feedback[m] + self.b[m]) - psi(self.sum_of_right_feedback[m] + self.sum_of_wrong_feedback[m] + self.a[m] + self.b[m]) # psi( \sum h^w + alpha ) - psi( \sum h^r + \sum h^w + alpha + beta )
                        Ce[m] = self.sum_of_right_feedback[m] / (self.sum_of_right_feedback[m] + self.sum_of_wrong_feedback[m]) # debug purpose only - not used in this agent
                
                if np.max(np.abs(Ce - Ce_old)) < 1e-3:
                    break; # if Ce does not change much, stop EM iterations

            # set the new Ce (avoid 0.0 and 1.0)
            self.Ce = np.clip(Ce, 0.001, 0.999)
            self.psi_for_hr = np.clip(self.psi_for_hr, -6.91, 0.0) # -6.91 = log(1e-3) (log of the smallest number)
            self.psi_for_hw = np.clip(self.psi_for_hw, -6.91, 0.0) # -6.91 = log(1e-3) (log of the smallest number)
        return action

    def tabQL_Cest_em(self, obs, rw, done, fb, type=2, update_Cest=False):
        """
        Tabular Q-learning with Policy shaping & consistency level (C) estimation
        The consistency level estimation is based on EM algorithm 
        """
        assert self.C_fixed == None, "C_fixed must be None for tabQL_Cest_em"
    
        # initialise Q
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions])    # initialise Q function (s,a)
            
            self.nTrainer = len(fb)
            self.hp = np.ones([self.nTrainer, self.nStates, self.nActions]) * 0.0
            self.hm = np.ones([self.nTrainer, self.nStates, self.nActions]) * 0.0
            self.Ce = np.ones(self.nTrainer) * 0.5
                       
        curr_state_idx = obs

        # Boltzmann exploration policy
        l_pr = self.Q[curr_state_idx,:]/self.tempConst
        
        # policy shaping
        self.d = self.hp - self.hm
        if type == 1:
            # type1 (general case)
            for trainerIdx in np.arange(self.nTrainer):
                for i in range(self.nActions):
                    l_pr[i] += self.d[trainerIdx,curr_state_idx,i] * np.log(self.Ce[trainerIdx]) \
                             - mylib.logadd(self.d[trainerIdx,curr_state_idx,i] * np.log(self.Ce[trainerIdx]), \
                                            self.d[trainerIdx,curr_state_idx,i] * np.log(1.0-self.Ce[trainerIdx]))
        else:
            # type2 (only one optimal action)
            for trainerIdx in np.arange(self.nTrainer):
                for i in range(self.nActions):
                    l_pr[i] += self.d[trainerIdx,curr_state_idx,i] * np.log(self.Ce[trainerIdx]) - self.d[trainerIdx,curr_state_idx,i] * np.log(1-self.Ce[trainerIdx])

        l_pr = l_pr - mylib.logsum(l_pr)
        pr = np.exp(l_pr)
        
        # decide action based on pr[] probability distribution
        action = np.min( np.where( np.random.rand() < np.cumsum(pr) ) )        

        
        # check if this is the first time...
        if self.prev_obs is not None:
            # one step TD algorithm
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            #td_err = (rw + self.gamma * np.sum(self.Q[curr_state_idx,:] * pr)) - self.Q[prev_state_idx, prev_action_idx] 
            max_a_idx = np.argmax(pr)
            if done:
                td_err = (rw) - self.Q[prev_state_idx, prev_action_idx]             
            else:
                td_err = (rw + self.gamma * self.Q[curr_state_idx, max_a_idx]) - self.Q[prev_state_idx, prev_action_idx] 
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
        
            # Human feedback updates
            self._collect_feedback(fb)
        
        # Update C estimations    
        if update_Cest:
            # prpare valid (s,a) pairs & s - at least one feedback
            sa_pairs = np.array([(s,a) for s in range(self.nStates) for a in range(self.nActions)])
            valid = np.ones(self.nStates * self.nActions) * False
            for n, (s,a) in enumerate(sa_pairs): 
                if np.sum(self.hp[:,s,a] + self.hm[:,s,a]) > 0:
                    valid[n] = True
            valid_sa_pairs = sa_pairs[valid==True]
            valid_s = np.unique([s for (s,a) in valid_sa_pairs])
            
            # prepare piror of Optimality flag O (P1/P0) from Q-learning
            ln_P_Q1 = np.zeros((self.nStates, self.nActions))
            ln_P_Q0 = np.zeros((self.nStates, self.nActions))
            for s in valid_s:
                # Boltzman exploration policy
                ln_pr = self.Q[s,:]/self.tempConst
                # normalise
                max_ln_pr = np.max(ln_pr)
                ln_pr = ln_pr - (max_ln_pr + np.log( np.sum( np.exp(ln_pr - max_ln_pr))))
                ln_P_Q1[s,:] = ln_pr
                ln_P_Q0[s,:] = np.log(1.0 - np.exp(ln_P_Q1[s,:]))
            
            d = self.hp - self.hm
            # Ce = np.ones(self.nTrainer) * 0.5 # 
            Ce = self.Ce # set start point of C (should start from 0.5?)
            ln_P1 = np.zeros((self.nStates, self.nActions))
            ln_P0 = np.zeros((self.nStates, self.nActions))
            for k in range(20): # EM iteration
                # E-step (compute posterior of O)
                if type == 1:
                    # type1 (general case)
                    for (s,a) in valid_sa_pairs:
                        ln_P1[s,a] = ln_P_Q1[s,a] + np.sum(d[:,s,a] * np.log(Ce))
                        ln_P0[s,a] = ln_P_Q0[s,a] + np.sum(d[:,s,a] * np.log(1.0-Ce))
                        ln_partition = mylib.logadd(ln_P0[s,a], ln_P1[s,a])
                        ln_P0[s,a] = ln_P0[s,a] - ln_partition
                        ln_P1[s,a] = ln_P1[s,a] - ln_partition
                else:
                    # type2 (only one optimal action)
                    for (s,a) in valid_sa_pairs:
                        ln_P1[s,a] = ln_P_Q1[s,a] + np.sum(d[:,s,a] * np.log(Ce)) \
                                                  - np.sum(d[:,s,a] * np.log(1.0-Ce))
                        ln_P0_ = -np.inf
                        for a_ in range(self.nActions):
                            if a_ != a:
                                ln_P0_ = mylib.logadd(ln_P0_, 
                                                    ln_P_Q1[s,a_]
                                                  + np.sum(d[:,s,a_] * np.log(Ce)) 
                                                  - np.sum(d[:,s,a_] * np.log(1.0-Ce)))
                        ln_P0[s,a] = ln_P0_
                        ln_partition = mylib.logadd(ln_P0[s,a], ln_P1[s,a])
                        ln_P0[s,a] = ln_P0[s,a] - ln_partition
                        ln_P1[s,a] = ln_P1[s,a] - ln_partition
                
                # M-step (compute C)
                Ce_old = Ce.copy()
                P1, P0 = np.exp(ln_P1), np.exp(ln_P0)
                for m in range(self.nTrainer):
                    if np.sum(self.hp[m,:,:] + self.hm[m,:,:]) > 0:
                        Ce[m] = np.sum(P1 * self.hp[m,:,:] + P0 * self.hm[m,:,:]) / np.sum(self.hp[m,:,:] + self.hm[m,:,:])
                
                Ce = np.clip(Ce, 0.001, 0.999) 
                
                if np.max(np.abs(Ce - Ce_old)) < 1e-3:
                    break; # if Ce does not change much, stop EM iterations

            # set the new Ce (avoid 0.0 and 1.0)
            self.Ce = np.clip(Ce, 0.001, 0.999)
        return action
    
    def tabQL_Cest_org(self, obs, rw, done, fb, type=2):
            # initialise Q
            if self.Q is None:
                #self.Q = np.ones([self.nStates, self.nActions])*50 # initialise Q function (s,a)
                self.Q = np.zeros([self.nStates, self.nActions])    # initialise Q function (s,a)
                
                self.nTrainer = len(fb)
                self.hp = np.ones([self.nTrainer, self.nStates, self.nActions]) * 0.1
                self.hm = np.ones([self.nTrainer, self.nStates, self.nActions]) * 0.1
                if self.C_fixed is None:
                    self.Ce = np.ones(self.nTrainer) * 0.5
                else:
                    self.Ce = self.C_fixed
                self.ave_nFBs = np.ones(self.nTrainer) * 1
                self.ave_absQ = np.ones(self.nTrainer) * 1

            curr_state_idx = obs

            """
            # Boltzmann exploration policy
            pr = np.exp(self.Q[curr_state_idx,:]/self.tempConst)
            pr[np.isinf(pr)] = np.finfo(np.float32).max # replace INF with max possible value with float32 type to avoid NaN in Pr
            
            # policy shaping
            self.d = self.hp - self.hm
            for i in range(self.nActions):
                pr[i] *= self.Ce ** self.d[curr_state_idx,i] * (1-self.Ce) ** sum(self.d[curr_state_idx,np.arange(self.nActions)!=i])
            pr[np.isinf(pr)] = np.finfo(np.float32).max # replace INF with max possible value with float32 type to avoid NaN in Pr        
            pr = pr / np.sum(pr)
            """
            # Boltzmann exploration policy
            l_pr = self.Q[curr_state_idx,:]/self.tempConst
            
            # policy shaping
            self.d = self.hp - self.hm
            if type == 1:
                # type1 (general case)
                for trainerIdx in np.arange(self.nTrainer):
                    for i in range(self.nActions):
                        l_pr[i] += self.d[trainerIdx,curr_state_idx,i] * np.log(self.Ce[trainerIdx]) \
                                 - mylib.logadd(self.d[trainerIdx,curr_state_idx,i] * np.log(self.Ce[trainerIdx]), \
                                                self.d[trainerIdx,curr_state_idx,i] * np.log(1.0-self.Ce[trainerIdx]))
            else:
                # type2 (only one optimal action)
                for trainerIdx in np.arange(self.nTrainer):
                    for i in range(self.nActions):
                        l_pr[i] += self.d[trainerIdx,curr_state_idx,i] * np.log(self.Ce[trainerIdx]) - self.d[trainerIdx,curr_state_idx,i] * np.log(1-self.Ce[trainerIdx])


            l_pr = l_pr - mylib.logsum(l_pr)
            pr = np.exp(l_pr)
            
            # decide action based on pr[] probability distribution
            action = np.min( np.where( np.random.rand() < np.cumsum(pr) ) )        
            
            # check if this is the first time...
            if self.prev_obs is not None:
                # one step TD algorithm
                prev_state_idx = self.prev_obs
                prev_action_idx = self.prev_act
                
                #td_err = (rw + self.gamma * np.sum(self.Q[curr_state_idx,:] * pr)) - self.Q[prev_state_idx, prev_action_idx] 
                a_idx = np.argmax(pr)
                if done:
                    td_err = rw - self.Q[prev_state_idx, prev_action_idx] 
                else:               
                    td_err = (rw + self.gamma * self.Q[curr_state_idx,a_idx]) - self.Q[prev_state_idx, prev_action_idx] 
                self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
            
                # Human feedback updates
                self._collect_feedback(fb)
                
                for trainerIdx in np.arange(self.nTrainer):
                    if self.C_fixed is None:
                        ret = self.estimateC(self.hp[trainerIdx,:,:], \
                                            self.hm[trainerIdx,:,:], \
                                            self.Ce[trainerIdx],
                                            self.ave_absQ[trainerIdx],
                                            self.ave_nFBs[trainerIdx], \
                                            type)
                        self.Ce[trainerIdx] = ret['Ce']
                        self.ave_nFBs[trainerIdx] = ret['ave_nFBs']
                        self.ave_absQ[trainerIdx] = ret['ave_absQ']
                        
                    else:
                        # fixed Cest case
                        self.Ce[trainerIdx] = self.C_fixed[trainerIdx] # <- edit fixed number here

            return action

    def estimateC(self, hp, hm, Ce, ave_absQ, ave_nFBs, type):
        l_pr = self.Q[self.prev_obs,:]/self.tempConst
        l_pr = mylib.lognorm(l_pr)
        pr = np.exp(l_pr)
        
        p1q = pr[self.prev_act]
        p0q = np.sum(pr[np.arange(len(pr)) != self.prev_act])        
        p1 = p1q
        p0 = p0q        
        C = 0.5
        
        d = hp - hm
        for n in range(50):
            # M-step
            Cnxt = (p1 * hp[self.prev_obs,self.prev_act] + p0 * hm[self.prev_obs,self.prev_act]) / (hp[self.prev_obs,self.prev_act] + hm[self.prev_obs,self.prev_act])
            Cnxt = np.round(Cnxt * 100) / 100
            if C == Cnxt:
                break
            else:
                if Cnxt == 1.0:
                    C = 1.0 - np.finfo(np.float32).resolution
                elif Cnxt == 0.0:
                    C = np.finfo(np.float32).resolution
                else:
                    C = Cnxt
                
            """ Linear Implementation
            # E-step
            p1_ = p1q * C**self.d[self.prev_obs,self.prev_act] * (1-C)**np.sum(self.d[self.prev_obs,np.arange(self.nActions)!=self.prev_act])
            if np.isinf(p1_):
                p1_ = np.finfo(np.float32).max
            
            p0_ = 0
            for i in np.arange(self.nActions):
                if i != self.prev_act:
                    p0_ += pr[i] * C**self.d[self.prev_obs, i] * (1-C)**np.sum(self.d[self.prev_obs,np.arange(self.nActions)!=i])
            if np.isinf(p0_):
                p0_ = np.finfo(np.float32).max
            p1 = p1_ / (p1_ + p0_)
            p0 = p0_ / (p1_ + p0_)
            """
            # E-step
            if type == 1:
                # type1 (general case)
                l_p1 = np.log(p1q) + d[self.prev_obs,self.prev_act] * np.log(C)     - mylib.logadd(d[self.prev_obs,self.prev_act] * np.log(C), d[self.prev_obs,self.prev_act] * np.log(1.0-C))
                l_p0 = np.log(p0q) + d[self.prev_obs,self.prev_act] * np.log(1.0-C) - mylib.logadd(d[self.prev_obs,self.prev_act] * np.log(C), d[self.prev_obs,self.prev_act] * np.log(1.0-C))
            else:
                # type2 (only one optimal action)
                l_p1 = np.log(p1q) + d[self.prev_obs,self.prev_act] * np.log(C) \
                                   + np.sum(d[self.prev_obs,np.arange(self.nActions)!=self.prev_act]) * np.log(1-C)
                l_p0 = -np.inf
                for i in np.arange(self.nActions):
                    if i != self.prev_act:
                        l_p0 = mylib.logadd(l_p0,  \
                                            np.log(pr[i])   \
                                                + d[self.prev_obs, i] * np.log(C)    \
                                                + np.sum(d[self.prev_obs,np.arange(self.nActions)!=i]) * np.log(1-C) )
            l_p1_ = l_p1 - mylib.logadd(l_p0, l_p1)
            l_p0_ = l_p0 - mylib.logadd(l_p0, l_p1)
            p1 = np.exp(l_p1_)
            p0 = np.exp(l_p0_)              
            
        nFBs = np.sum(hp[self.prev_obs,:]) + \
                        np.sum(hm[self.prev_obs,:])
        absQ = np.sum(np.abs(self.Q[self.prev_obs,:]))
            
        # average C over (s,a)
        lr = nFBs*absQ / (ave_nFBs*ave_absQ) / 16
        lr = np.min([lr, 1])
        #lr = 1/32 # FIXED LR #
        Ce = Ce + (C - Ce) * lr

        ave_nFBs = ave_nFBs + (nFBs - ave_nFBs) * lr
        ave_absQ = ave_absQ + (absQ - ave_absQ) * lr
        
        return {'Ce':Ce, 'ave_nFBs':ave_nFBs, 'ave_absQ':ave_absQ}

    def save(self, fname):
        with open(fname + '.pkl', 'wb') as fid:
            pickle.dump([self.Q, self.d], fid)
        return
    
    def load(self, fname):
        with open(fname, 'rb') as fid:
            Q, d = pickle.load(fid)
            self.Q = Q
            if d is not []:
                self.d = d
        return
    
    def tabQL_IPL(self, obs, rw, done, fb, type=1, update_Cest=None):
        """
        Inverse Preference Learning - learns directly from preferences without reward model
        """
        # Initialize Q
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions])
            self.preference_counts = np.zeros([self.nStates, self.nActions])
        
        curr_state_idx = obs
        
        # Collect preference feedback and update preference counts
        if fb:
            for trainer_fb in fb:
                for feedback in trainer_fb:
                    if hasattr(feedback, 'good_actions'):
                        for a in feedback.good_actions:
                            self.preference_counts[feedback.state, int(a)] += 1
                    if hasattr(feedback, 'bad_actions'):
                        for a in feedback.bad_actions:
                            self.preference_counts[feedback.state, int(a)] -= 1
        
        # Action selection based on preference-weighted Q-values
        if np.sum(np.abs(self.preference_counts[curr_state_idx, :])) > 0:
            # Use preference information
            pref_weights = self.preference_counts[curr_state_idx, :] + 1e-6  # avoid zeros
            weighted_q = self.Q[curr_state_idx, :] + 0.1 * pref_weights
            action = np.argmax(weighted_q)
        else:
            # Standard greedy action
            action = np.argmax(self.Q[curr_state_idx, :])
        
        # Q-learning update
        if self.prev_obs is not None:
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            # Use preference signal as reward proxy
            pref_reward = np.mean([np.sum(self.preference_counts[prev_state_idx, :]) for _ in fb]) if fb else 0
            td_err = (rw + pref_reward + self.gamma * np.max(self.Q[curr_state_idx, :])) - self.Q[prev_state_idx, prev_action_idx]
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
        
        return action

    def tabQL_visit_active(self, obs, rw, done, fb, type=1, update_Cest=None):
        """
        Visit-count based active learning - prioritizes least visited state-action pairs
        """
        # Initialize Q and visit counts
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions])
            self.visit_counts = np.zeros([self.nStates, self.nActions])
        
        curr_state_idx = obs
        
        # Update visit count
        if self.prev_obs is not None and self.prev_act is not None:
            self.visit_counts[self.prev_obs, self.prev_act] += 1
        
        # Action selection: epsilon-greedy with visit-based exploration
        if np.random.rand() < self.eps:
            # Choose least visited action in current state
            action = np.argmin(self.visit_counts[curr_state_idx, :])
        else:
            # Greedy action selection
            action = np.argmax(self.Q[curr_state_idx, :])
        
        # Standard Q-learning update with feedback shaping
        if self.prev_obs is not None:
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            # Standard TD update
            td_err = (rw + self.gamma * np.max(self.Q[curr_state_idx, :])) - self.Q[prev_state_idx, prev_action_idx]
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
            
            # Apply feedback shaping if available
            if fb:
                self._collect_feedback(fb)
                feedback_bonus = 0
                for trainer_idx in range(len(fb)):
                    feedback_bonus += (self.hp[trainer_idx, prev_state_idx, prev_action_idx] - 
                                    self.hm[trainer_idx, prev_state_idx, prev_action_idx]) * 0.1
                self.Q[prev_state_idx, prev_action_idx] += feedback_bonus
        
        return action

    def tabQL_majority_vote(self, obs, rw, done, fb, type=1, update_Cest=None):
        """
        Simple majority vote aggregation across crowd workers
        """
        # Initialize Q and feedback aggregation
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions])
            self.nTrainer = len(fb) if fb else 1
            self.hp = np.zeros([self.nTrainer, self.nStates, self.nActions])
            self.hm = np.zeros([self.nTrainer, self.nStates, self.nActions])
            self.majority_weights = np.ones([self.nStates, self.nActions])
        
        curr_state_idx = obs
        
        # Collect feedback and compute majority vote
        if fb:
            self._collect_feedback(fb)
            
            # Simple majority voting per state-action pair
            for s in range(self.nStates):
                for a in range(self.nActions):
                    pos_votes = np.sum(self.hp[:, s, a])
                    neg_votes = np.sum(self.hm[:, s, a])
                    if pos_votes + neg_votes > 0:
                        self.majority_weights[s, a] = (pos_votes - neg_votes) / (pos_votes + neg_votes)
        
        # Action selection with majority vote weighting
        weighted_q = self.Q[curr_state_idx, :] + 0.1 * self.majority_weights[curr_state_idx, :]
        action = np.argmax(weighted_q)
        
        # Standard Q-learning update
        if self.prev_obs is not None:
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            td_err = (rw + self.gamma * np.max(self.Q[curr_state_idx, :])) - self.Q[prev_state_idx, prev_action_idx]
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
            
            # Apply majority vote shaping
            majority_bonus = self.majority_weights[prev_state_idx, prev_action_idx] * 0.05
            self.Q[prev_state_idx, prev_action_idx] += majority_bonus
        
        return action

    def tabQL_uncertainty_active(self, obs, rw, done, fb, type=1, update_Cest=None):
        """
        Uncertainty-based active learning using Q-value variance
        """
        # Initialize Q and uncertainty tracking
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions])
            self.Q_variance = np.ones([self.nStates, self.nActions])  # Initialize with high uncertainty
            self.update_counts = np.zeros([self.nStates, self.nActions])
        
        curr_state_idx = obs
        
        # Action selection based on uncertainty
        if np.random.rand() < self.eps:
            # Choose action with highest uncertainty
            action = np.argmax(self.Q_variance[curr_state_idx, :])
        else:
            # Greedy action selection
            action = np.argmax(self.Q[curr_state_idx, :])
        
        # Q-learning update with uncertainty tracking
        if self.prev_obs is not None:
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            # Compute TD error
            td_err = (rw + self.gamma * np.max(self.Q[curr_state_idx, :])) - self.Q[prev_state_idx, prev_action_idx]
            
            # Update Q-value
            old_q = self.Q[prev_state_idx, prev_action_idx]
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
            
            # Update uncertainty (variance) using online variance formula
            self.update_counts[prev_state_idx, prev_action_idx] += 1
            n = self.update_counts[prev_state_idx, prev_action_idx]
            if n > 1:
                delta = abs(td_err)
                self.Q_variance[prev_state_idx, prev_action_idx] = ((n-1) * self.Q_variance[prev_state_idx, prev_action_idx] + delta**2) / n
            
            # Apply feedback if available
            if fb:
                self._collect_feedback(fb)
        
        return action

    def tabQL_dawid_skene(self, obs, rw, done, fb, type=1, update_Cest=None):
        """
        Dawid-Skene algorithm for crowd learning with reliability estimation
        """
        # Initialize Q and Dawid-Skene parameters
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions])
            if fb:
                self.nTrainer = len(fb)
                # Worker reliability matrix: [worker, true_label, reported_label]
                self.worker_reliability = np.ones([self.nTrainer, 2, 2]) * 0.5
                self.worker_reliability[:, 0, 0] = 0.7  # True negative correctly identified
                self.worker_reliability[:, 1, 1] = 0.7  # True positive correctly identified
                self.hp = np.zeros([self.nTrainer, self.nStates, self.nActions])
                self.hm = np.zeros([self.nTrainer, self.nStates, self.nActions])
        
        curr_state_idx = obs
        
        # Collect feedback and estimate true labels using Dawid-Skene
        if fb and update_Cest:
            self._collect_feedback(fb)
            self._update_dawid_skene()
        
        # Action selection using reliability-weighted feedback
        if hasattr(self, 'reliable_feedback'):
            weighted_q = self.Q[curr_state_idx, :] + 0.1 * self.reliable_feedback[curr_state_idx, :]
            action = np.argmax(weighted_q)
        else:
            action = np.argmax(self.Q[curr_state_idx, :])
        
        # Standard Q-learning update
        if self.prev_obs is not None:
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            td_err = (rw + self.gamma * np.max(self.Q[curr_state_idx, :])) - self.Q[prev_state_idx, prev_action_idx]
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
        
        return action

    def _update_dawid_skene(self):
        """
        Helper method to update Dawid-Skene reliability estimates
        """
        if not hasattr(self, 'reliable_feedback'):
            self.reliable_feedback = np.zeros([self.nStates, self.nActions])
        
        # Simplified Dawid-Skene update
        for s in range(self.nStates):
            for a in range(self.nActions):
                if np.sum(self.hp[:, s, a] + self.hm[:, s, a]) > 0:
                    # Estimate true label based on worker reliability
                    weighted_pos = 0
                    weighted_neg = 0
                    for w in range(self.nTrainer):
                        reliability = np.mean([self.worker_reliability[w, 0, 0], self.worker_reliability[w, 1, 1]])
                        weighted_pos += self.hp[w, s, a] * reliability
                        weighted_neg += self.hm[w, s, a] * reliability
                    
                    if weighted_pos + weighted_neg > 0:
                        self.reliable_feedback[s, a] = (weighted_pos - weighted_neg) / (weighted_pos + weighted_neg)