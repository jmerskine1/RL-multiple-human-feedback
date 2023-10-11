import jax
import jax.numpy as np
from tqdm import tqdm
import envPacMan
from agent import agent

from config import setup, environment, parameters
from library.utilities import RLmon, save_results
from library.learning_functions import tabQL_ps, tabQLgreedy

        
# -----------------------------------------------------

def main():
    
    dispON = setup['dispON']
    print(f"start--{setup['algID']} {setup['simInfo']}")
    
    # prepare RL monitor module
    monitors = {
        'aveRW'   :RLmon(1),
        'aveC'    :RLmon(len(setup['C'])),
        'Alpha'   :RLmon(len(setup['C'])),
        'Beta'    :RLmon(len(setup['C']))
                }
    
    pac_env = envPacMan.environment()
       
    for k in range(setup['trial_count']):
        print('trial: {0}'.format(k))
        
        pac_env.reset()
        
        agent_h  = agent(tabQL_ps)
        
        # Setup ORACLE
        oracle_h = agent(tabQLgreedy)
        oracle_h.load('learnedStates/pacman_tabQL_oracle.pkl')   # load pre-learned Q function    
        oracle_h.alpha = 0                          # set learning rate to zero (no learning)    

        action_list = list(environment['actions'].keys())
        action = 0                    # default action
        obs = pac_env.st2ob()          # observation
        rw = 0                        # reward
        totRW = 0                     # total reward in this episode
        done = False                  # episode completion flag
        fb = np.ones(len(setup['C'])) * np.full((), -np.inf) # Human feedback
        
        for i in tqdm(range(setup['episode_count'])):
            
            for j in range(setup['max_steps']):
                
                if dispON:

                    print('action:{0}'.format(action_list[action]))
                    pac_env.display()    
                    tmp = input('>>')
                    if tmp == 'stop':
                        dispON = False
        
                # call agent
                action = agent_h.act(obs, fb, rw, done)
                
                if done:
                    agent_h.Cest_em()

                # call oracle to get 'right' action
                rightAction = oracle_h.act(obs, fb, rw, done)
                    
                # call environment
                obs, rw, done = pac_env.step(action)
                
                # 'human' feedback generation (by using ORACLE)
                for trainerIdx in np.arange(len(fb)):
                    if jax.random.uniform(parameters['key']) < setup['L'][trainerIdx]:
                        if jax.random.uniform(parameters['key']) < setup['C'][trainerIdx]:
                            fb = fb.at[trainerIdx].set(np.array_equal(action, rightAction))
                            # fb[trainerIdx] = (action == rigßhtAction)     # Right feedback
                        else:
                            fb = fb.at[trainerIdx].set(not(np.array_equal(action, rightAction)))
                            # fb[trainerIdx] = not (action == rightAction) # Wrong feedback
                    else:
                        fb[trainerIdx] = np.NaN # no feedback
                
                # accumrate total reward
                totRW += rw
                
                # if done==True, call agent once more to learn 
                # the final transition, then finish this episode.
                if done:
                    agent_h.act(obs, fb, rw, done)
                    break
            
            if i % 100 == 0:
                print(f"{k}, {i}: Ce: {agent_h.Ce} \t total reward: {totRW}")
            
            # store result
            monitors['aveRW'].store(i, k, totRW)
            monitors['aveC'].store(i, k, agent_h.Ce)
            if hasattr(agent_h, 'sum_of_right_feedback'):
                # store VI algorithm parameters
                monitors['Alpha'].store(i, k, agent_h.sum_of_right_feedback + agent_h.a)
                monitors['Beta'].store(i, k,  agent_h.sum_of_wrong_feedback + agent_h.b)
            
            # Reset environment
            pac_env.reset()
            agent_h.prev_obs = None
            obs = pac_env.st2ob()
            rw = 0
            totRW = 0
            done = False
                        
        # Clear agent class except the last trial
        if k < setup['trial_count']-1:
            del agent_h
            del oracle_h

    # Save results
    save_results(monitors,agent_h)

if __name__ == '__main__':
    main()
