# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:43:24 2018

@author: taku.yamagata

Agent class
"""
import numpy as np
import pickle

from math import gamma
from math import sqrt
from scipy.stats import t

import mylib

class agent(object):

    def __init__(self, algID, nStates, nActions):
        self.reset(algID, nStates, nActions)
        return
        
        
    def reset(self, algID, nStates, nActions):
        self.gamma    = 0.9 # discount factor
        self.alpha    = 0.05 #1.0/128.0 # learning rate
        self.eps      = 0.1 # e-greedy policy parametre (probability to select exploration action)
        
        self.Q        = []           # value function - Q
        self.prev_obs = np.NaN           # previous observation
        self.prev_act = []           # previous action
        
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
        return
    
    
    def act(self, a, obs, rw, done, fb, C, skip_confidence=False):

        # action = 0 # default action.

        # Value function estimation        
        if self.algID == 'tabQL_ps_Cest':
            action = self.tabQL_ps_Cest(obs, rw, fb, skip_confidence=skip_confidence)
        elif self.algID == 'tabQLgreedy':
            action = self.tabQLgreedy(obs, rw)
        
        else:
            raise ValueError('Invalid algorithm option ID')
        
        # store the current observation/action for the next call
        self.prev_obs = obs
        self.prev_act = action

        return action

    # Tabular one step Temporal Difference
    def tabQLgreedy(self, obs, rw):
       
        # initialise Q
        if len(self.Q) == 0:
            #self.Q = np.ones([5, 2])*20            # initialise Q function (s,a)
            self.Q = np.zeros([self.nStates, self.nActions]) # initialise Q function (s,a)
            #self.Q[4,:] = 0
        
        curr_state_idx = obs
        
        # check if this is the first time...
        if self.prev_obs != []:
            # one step TD algorithm
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            td_err = (rw + self.gamma * max(self.Q[curr_state_idx,:])) - self.Q[prev_state_idx, prev_action_idx] 
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
            
        # Greedy policy (always select the best action)
        action = np.argmax(self.Q[curr_state_idx,:])

        return action

    def tabQL_ps_Cest(self, obs, rw, fb, skip_confidence=False):
        # initialise Q
        if len(self.Q) == 0:
            #self.Q = np.ones([self.nStates, self.nActions])*50 # initialise Q function (s,a)
            self.Q = np.zeros([self.nStates, self.nActions])    # initialise Q function (s,a)
            
            self.nTrainer = len(fb)
            self.hp = np.ones([self.nTrainer, self.nStates, self.nActions]) * 0.1
            self.hm = np.ones([self.nTrainer, self.nStates, self.nActions]) * 0.1
            self.Ce = np.ones(self.nTrainer) * 0.5
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
        for trainerIdx in np.arange(self.nTrainer):
            for i in range(self.nActions):
                l_pr[i] += self.d[trainerIdx,curr_state_idx,i] * np.log(self.Ce[trainerIdx]) + sum(self.d[trainerIdx,curr_state_idx,np.arange(self.nActions)!=i]) * np.log(1-self.Ce[trainerIdx])

            max_l_pr = np.max(l_pr)
            l_pr = l_pr - (max_l_pr + np.log( np.sum( np.exp(l_pr - max_l_pr))))
            
        pr = np.exp(l_pr)
        
        # decide action based on pr[] probability distribution
        action = np.min( np.where( np.random.rand() < np.cumsum(pr) ) )        
    
        if not skip_confidence:
            # check if this is the first time...
            if self.prev_obs != np.NaN:
                # one step TD algorithm
                prev_state_idx = self.prev_obs
                prev_action_idx = self.prev_act
                
                #td_err = (rw + self.gamma * np.sum(self.Q[curr_state_idx,:] * pr)) - self.Q[prev_state_idx, prev_action_idx] 
                a_idx = np.argmax(pr)
                td_err = (rw + self.gamma * self.Q[curr_state_idx,a_idx]) - self.Q[prev_state_idx, prev_action_idx] 
                self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err


                # Human feedback updates
                for trainerIdx in np.arange(self.nTrainer):
                    if fb[trainerIdx] == True:
                        self.hp[trainerIdx, prev_state_idx, prev_action_idx] += 1
                    elif fb[trainerIdx] == False:
                        self.hm[trainerIdx, prev_state_idx, prev_action_idx] += 1
                    
                    Cest_enable = True # hyper param to enable Cest
                    if Cest_enable:
                        ret = self.estimateC(self.hp[trainerIdx,:,:], \
                                            self.hm[trainerIdx,:,:], \
                                            self.Ce[trainerIdx],
                                            self.ave_absQ[trainerIdx],
                                            self.ave_nFBs[trainerIdx])
                        self.Ce[trainerIdx] = ret['Ce']
                        self.ave_nFBs[trainerIdx] = ret['ave_nFBs']
                        self.ave_absQ[trainerIdx] = ret['ave_absQ']
                    else:
                        # fixed Cest case
                        self.Ce[trainerIdx] = 0.8 # <- edit fixed number here

        return action

    def estimateC(self, hp, hm, Ce, ave_absQ, ave_nFBs):
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
            if d!=[]:
                self.d = d
        return
    
