import numpy as np

import envPacMan as environment 
from agent import agent
from RLmon import RLmon
from feedback import *

# ==================================================================================================
def main(algID   = 'tabQLgreedy',  # Agent Algorithm   'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2', 
                                        #                   'tabQL_Cest_em_t1', 'tabQL_Cest_em_org_t2', 
                                        #                   'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo = '_nochase_',              # Filename header
         env_size = 'small',            # Pacman environment size 'small' or 'medium'
         episode_count = 50_000,       # number of episodes to learn
         max_steps = 500,               # max. number of steps in a episode
         ):

    print(f"start--{algID} {simInfo}")
    dispON = False
    
    # prepare RL monitor module    
    monitor = RLmon(['return'])
    
    env_h = environment.env(env_size)            
    
    env_h.reset()
    agent_h  = agent(algID, env_h.nStates(), len(env_h.action_list()))
    agent_h.tempConst = 100 # default value is 1.5
    
    action_list = env_h.action_list()
    action = 0 
    ob = env_h.st2ob()            # observation
    rw = 0                        # reward
    totRW = 0                     # total reward in this episode
    done = False                  # episode completion flag
    fb = [[]] # Human feedback
    
    totalRW_list = []
    
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
            action = agent_h.act(action, ob, rw, done, fb, 0.5, update_Cest=False)
                
            # call environment
            ob, rw, done = env_h.step(action_list[action])

            # accumrate total reward
            totRW += rw

            if done or j == max_steps - 1:             
                agent_h.act(action, ob, rw, done, fb, np.array([]), update_Cest=False)
                break
        
        totalRW_list.append(totRW)
        if i % 1_000 == 0:
            print(f"{i}:\t total reward: average: {np.mean(totalRW_list):.1f}, min: {np.min(totalRW_list):.1f}, max: {np.max(totalRW_list):.1f}")
            totalRW_list = []
        
        # store result
        monitor.store('return', totRW)

        # Reset environment
        env_h.reset(random=False, pellet_random=False)
        agent_h.prev_obs = None
        ob = env_h.st2ob()
        rw = 0
        totRW = 0
        done = False
    
        if i % 500_000 == 0:
            # save model
            agent_h.save(f'learnedStates/{str(simInfo)}_{i}')
    
    # store the accumurated results
    monitor.store(done=True)

    # Save results
    fname = 'results/results_' + 'Env_' + str(env_size) + '_' + str(algID) + str(simInfo)
    monitor.saveData(fname)
        
    agent_h.save(f'learnedStates/{str(simInfo)+ str(env_size)+str(algID)}')



if __name__ == '__main__':
    main()
