import numpy as np

import envPacMan as environment 
from agent import agent
from RLmon import RLmon
from feedback import *
import pickle

# ==================================================================================================
def main(algID   =  'tabQL_Cest_vi_t2',  # Agent Algorithm   'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2', 
                                        #                   'tabQL_Cest_em_t1', 'tabQL_Cest_em_org_t2', 
                                        #                   'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2', 'tabQLrandom'
         feedback_type = 'ranked-feedback', # feedback type 'binary-feedback', 'soft-feedback', 'crisp-set', 'soft-set', 'ranked-feedback
         simInfo = '_50trials_10fb_ranked',              # Filename header
         env_size = 'small',            # Pacman environment size 'small' or 'medium'
         trial_count = 1,             # number of learning trial
         episode_count = 1000,          # number of episodes to learn
         max_steps = 500,               # max. number of steps in a episode
         L  = np.array([10,10,10,10]),          # probability to give a feedback
         C  = np.array([0.9,0.8,0.6,0.3]),          # a = [1Human feedback confidence level
         prior_alpha  = 1.0,            # alpha for C prior
         prior_beta   = 1.0,            # beta  for C prior
         no_reward = False,             # agent learns the policy without reward (feedback only)
         C_fixed = None,                # None: learn C, np.array(): fixed C (fixed C only works with "tabQL_Cest_em_org_t1" or "tabQL_Cest_em_org_t2")
         update_Cest_interval = 5,      # Cest update interval (number of espisodes)
         active_feedback_type = 'count', #'count',   # active feedback type (None: no active feedback, 'count')
         reset_env_random = False,      # environment start with random state (True: random, False: fixed)
         randomise_trajectory = False,  # overwrite trajectory data with random data (True: overwrite, False: do not overwrite)
         oracle4fb_type = 'old',        # oracle type for generating feedback ('old' or 'new')
         oracle4al_type = 'old',        # oracle type for active learning     ('old' or 'new')
         ):

    print(f"start--{algID} {simInfo}")
    dispON = False
    
    # prepare RL monitor module
    legendStr = []
    for n in range(len(C)):
        legendStr.append('L={0},C={1}'.format(L[n], C[n]))
    
    monitor = RLmon(['return', 'Cest', 'alpha', 'beta'])
    trajectory = Trajectory()
    all_feedback = []
    
    env_h = environment.env(env_size)            
    for k in range(trial_count):
        print('trial: {0}'.format(k))
        
        env_h.reset()
        agent_h  = agent(algID, env_h.nStates(), len(env_h.action_list()), 
                         a=prior_alpha, b=prior_beta, 
                         C_fixed=C_fixed, 
                        )
        
        # Setup ORACLE
        oracle_h = agent('tabQLgreedy', env_h.nStates(), len(env_h.action_list()))
        if env_size == 'small': # load pre-learned Q function
            if oracle4fb_type == 'new': 
                oracle_h.load('learnedStates/pacman_small_tabQL_oracle_randomised_env.pkl')
            else:
                oracle_h.load('learnedStates/_nochase_smalltabQLgreedy.pkl')

        elif env_size == 'medium':
            oracle_h.load('/Users/jonathanerskine/University of Bristol/RL-multiple-human-feedback/learnedStates/_randenv_mediumtabQLgreedy.pkl')
        
        elif env_size == 'medium_sparse':
            oracle_h.load('/Users/jonathanerskine/University of Bristol/RL-multiple-human-feedback/learnedStates/_randenv_medium_sparsetabQLgreedy.pkl')
            # oracle_h.load('learnedStates/pacman_medium_tabQL_oracle.pkl')
        else:
            raise ValueError(f"nvalid env_size value - must be 'small' or 'medium': {env_size}")
        oracle_h.alpha = 0                          # set learning rate to zero (no learning)

        oracle_h_act = oracle_h
        # oracle_h_act = agent('tabQLgreedy', env_h.nStates(), len(env_h.action_list()))
        # if env_size == 'small': # load pre-learned Q function
        #     if oracle4al_type == 'new': 
        #         oracle_h_act.load('learnedStates/pacman_small_tabQL_oracle_randomised_env.pkl')
        #     else:
        #         oracle_h_act.load('learnedStates/pacman_tabQL_oracle.pkl') 
        # elif env_size == 'medium':
        #     oracle_h_act.load('learnedStates/pacman_medium_tabQL_oracle.pkl')
        # else:
        #     raise ValueError(f"nvalid env_size value - must be 'small' or 'medium': {env_size}")
        # oracle_h.alpha = 0                          # set learning rate to zero (no learning)


        action_list = env_h.action_list()
        action = 0 
        ob = env_h.st2ob()            # observation
        rw = 0                        # reward
        totRW = 0                     # total reward in this episode
        done = False                  # episode completion flag
        fb = [[] for n in range(len(C))] # Human feedback
        update_Cest = False

        totalRW_list = []
        ob_for_feedback = None
        rightAction = None
        fbs = []
        
        for i in range(episode_count):

            trajectory.reset() # store trajectory for generating active feedback (generate feedback at the end of the episode)
            fb = [[] for n in range(len(C))] # Human feedback
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
                # call oracle to get 'right' action
                if np.any(L > 0.0):
                    rightAction = oracle_h.act(action, ob, rw, done, fb, C)
                    ob_for_feedback = ob
                    
                # call environment
                ob, rw, done = env_h.step(action_list[action])

                # accumrate total reward
                totRW += rw

                # store the trajectory for generating active feedback (generate feedback at the end of the episode)
                if np.any(L > 0.0):
                    s = np.random.randint(0, env_h.nStates()) if randomise_trajectory else ob_for_feedback
                    a = np.random.randint(0, len(env_h.action_list())) if randomise_trajectory else action
                    # a = np.argmax(agent_h.Q[s,:]) if randomise_trajecotry else action
                    opt_a = oracle_h.act(a, s, 0, False, [], []) if randomise_trajectory else rightAction
                    trajectory.append(state=s, action=a, optimal_action=opt_a, reward=rw, done=done)

                # set reward zero when simulating without reward scase
                if no_reward:
                    rw = 0.0
                
                # 'human' feedback generation (by using ORACLE)
                if np.any(L > 0.0):
                    # generate feedbacks
                    fb = generate_feedback(
                            trajectory=trajectory,
                            C=C,
                            L=L,
                            action_list=env_h.action_list(), 
                            agent_h=agent_h, 
                            oracle_h=oracle_h,
                            oracle_h_act = oracle_h_act, 
                            end_of_episode=(done or j == max_steps - 1), 
                            no_reward=no_reward, 
                            feedback_type = feedback_type, 
                            active_feedback_type=active_feedback_type,)
                               
                if done or j == max_steps - 1:                    
                    update_Cest = ((i+1) % update_Cest_interval == 0)
                    agent_h.act(action, ob, rw, done, fb, C, update_Cest=update_Cest)
                    agent_h.prev_obs = None
                    fbs.append(fb)
                    break
            

            totalRW_list.append(totRW)
            if i % 20 == 0:
                print(f"{k}, {i}: Ce: {agent_h.Ce} \t total reward: mean:{np.mean(totalRW_list):+.1f} (5 percentile: {np.percentile(totalRW_list, 5):+.1f}, 95 percentile: {np.percentile(totalRW_list, 95):+.1f})")
                totalRW_list = []
            
            # store result
            monitor.store('return', totRW)
            monitor.store('Cest', agent_h.Ce)
            if hasattr(agent_h, 'sum_of_right_feedback'):
                # store VI algorithm parameters
                monitor.store('alpha', agent_h.sum_of_right_feedback + agent_h.a)
                monitor.store('beta',  agent_h.sum_of_wrong_feedback + agent_h.b)
                
            # Reset environment
            env_h.reset(random=reset_env_random, pellet_random=reset_env_random)
            agent_h.prev_obs = None
            ob = env_h.st2ob()
            rw = 0
            totRW = 0
            done = False
        
        # save model
        # agent_h.save('learnedStates/pacman_medium_tabQL_oracle')
                        
        # Clear agent class except the last trial
        if k < trial_count-1:
            del agent_h
            del oracle_h
        
        all_feedback.append(fbs)
        
        # store the accumurated results
        monitor.store(done=True)

    # Save results
    fname = 'results/results_' + 'Env_' + str(env_size) + '_' + str(algID) + str(simInfo) + str(active_feedback_type)
    monitor.saveData(fname)

    with open(f'results/'+'feedback_Env_' + str(env_size) + '_' + str(algID) + str(simInfo) + str(active_feedback_type)+'.pkl','wb') as f:
        pickle.dump(all_feedback,f)
        
    #fname = 'results/plot_' + str(algID) + str(simInfo)
    #mon.savePlot(fname)
    
    # agent_h.save('learnedStates/pacman_' + str(algID))

if __name__ == '__main__':
    main()
