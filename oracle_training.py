import jax
import jax.numpy as np
from tqdm import tqdm
import envPacMan
from agent import agent

from config import setup, environment, parameters
from library.utilities import RLmon, save_results,dict_search
from library.learning_functions import tabQL_ps, tabQLgreedy

        
# -----------------------------------------------------
@profile
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
        
        g_agent  = agent(tabQLgreedy)
        

        
        action = np.array([1,0])      # default action
        obs = pac_env.st2ob()         # observation
        rw = 0                        # reward
        totRW = 0                     # total reward in this episode
        done = False                  # episode completion flag
        fb = None
        
        for i in tqdm(range(setup['episode_count'])):
            
            for j in range(setup['max_steps']):
                
                if dispON:
                    print('action:{0}'.format(dict_search(environment['actions'],action)))
                    pac_env.display()    
                    tmp = input('>>')
                    if tmp == 'stop':
                        dispON = False
        
                # call agent
                action = g_agent.act(obs, fb, rw, done)
                # g_agent.load('learnedStates/pacman_test.pkl')   # load pre-learned Q function    
                # g_agent.alpha = 0                          # set learning rate to zero (no learning)

                # call environment
                obs, rw, done = pac_env.step(action)
                # obs, rw, done = pac_env.step(np.array([1,0]))

                
                # accumrate total reward
                totRW += rw
                
                # if done==True, call agent once more to learn 
                # the final transition, then finish this episode.
                if done:
                    g_agent.act(obs, fb, rw, done)
                    break
            
            if i % 100 == 0:
                print(f"{k}, {i}: Ce: {g_agent.Ce} \t total reward: {totRW}")
            
            # store result
            monitors['aveRW'].store(i, k, totRW)
            monitors['aveC'].store(i, k, g_agent.Ce)
            if hasattr(g_agent, 'sum_of_right_feedback'):
                # store VI algorithm parameters
                monitors['Alpha'].store(i, k, g_agent.sum_of_right_feedback + g_agent.a)
                monitors['Beta'].store(i, k,  g_agent.sum_of_wrong_feedback + g_agent.b)
            
            # Reset environment
            pac_env.reset()
            g_agent.prev_obs = None
            obs = pac_env.st2ob()
            rw = 0
            totRW = 0
            done = False
                        
        # Clear agent class except the last trial
        if k < setup['trial_count']-1:
            del g_agent

    # Save results
    save_results(monitors,g_agent)

if __name__ == '__main__':
    main()
