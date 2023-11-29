import numpy as np

import envPacMan as environment 
from agent import agent

from RLmon import RLmon
        
# -----------------------------------------------------
def main(algID   = 'tabQL_ps_Cest',     # Agent Algorithm
         simInfo = '_tmp',              # Filename header
         trial_count = 200,             # number of learning trial
         episode_count = 2000,          # number of episodes to learn
         max_steps = 500,               # max. number of steps in a episode
         L  = np.array([1.0]),          # probability to give a feedback
         C  = np.array([0.2])           # Human feedback confidence level
         ):

    print("start---")
    dispON = False
    
    # prepare RL monitor module
    legendStr = []
    for n in range(len(C)):
        legendStr.append('L={0},C={1}'.format(L[n], C[n]))
    mon = RLmon(episode_count, 1)
    monC = RLmon(episode_count, len(C))
    
    env_h = environment.env()            
    for k in range(trial_count):
        print('trial: {0}'.format(k))
        
        env_h.reset()
        agent_h  = agent(algID, env_h.nStates(), len(env_h.action_list()))
        
        # Setup ORACLE
        oracle_h = agent('tabQLgreedy', env_h.nStates(), len(env_h.action_list()))
        oracle_h.load('learnedStates/pacman_tabQL_oracle.pkl')   # load pre-learned Q function    
        oracle_h.alpha = 0                          # set learning rate to zero (no learning)
        
        action_list = env_h.action_list()
        action = 0 
        ob = env_h.st2ob()            # observation
        rw = 0                        # reward
        totRW = 0                     # total reward in this episode
        done = False                  # episode completion flag
        fb = np.ones(len(C)) * np.NaN # Human feedback
        
        for i in range(episode_count):
            
            for j in range(max_steps):
                
                if dispON:
                    print('action:{0}'.format(action_list[action]))
                    d = env_h.display()
                    for n in range(len(d)):
                        print(d[n])
                        
                    tmp = input('>>')
                    if tmp == 'stop':
                        dispON = False

                # call agent
                action = agent_h.act(action, ob, rw, done, fb, 0.5)
                # call oracle to get 'right' action
                rightAction = oracle_h.act(action, ob, rw, done, fb, C)
                
                # call environment
                ob, rw, done = env_h.step(action_list[action])
                
                # 'human' feedback generation (by using ORACLE)
                for trainerIdx in np.arange(len(fb)):
                    if np.random.rand() < L[trainerIdx]:
                        if np.random.rand() < C[trainerIdx]:
                            fb[trainerIdx] = (action == rightAction)     # Right feedback
                        else:
                            fb[trainerIdx] = not (action == rightAction) # Wrong feedback
                    else:
                        fb[trainerIdx] = np.NaN # no feedback
                
                # accumrate total reward
                totRW += rw
                
                # if done==True, call agent once more to learn 
                # the final transition, then finish this episode.
                if done:
                    agent_h.act(action, ob, rw, done, fb, C)
                    break
            
            # store result
            mon.store(i, k, totRW)
            monC.store(i, k, agent_h.Ce)
            
            # Reset environment
            env_h.reset()
            agent_h.prev_obs = []
            ob = env_h.st2ob()
            rw = 0
            totRW = 0
            done = False
                        
        # Clear agent class except the last trial
        if k < trial_count-1:
            del agent_h
            del oracle_h

    # Save results
    fname = 'results/aveRW_' + str(algID) + str(simInfo)
    mon.saveData(fname)
    fname = 'results/aveC_' + str(algID) + str(simInfo)
    monC.saveData(fname)
    #fname = 'results/plot_' + str(algID) + str(simInfo)
    #mon.savePlot(fname)
    
    agent_h.save('learnedStates/pacman_' + str(algID))

if __name__ == '__main__':
    main()
