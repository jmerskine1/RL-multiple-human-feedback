import numpy as np
import envPacMan as environment 
from agent import agent
from RLmon import RLmon
from feedback import *
from trainer import PacmanTrainer
import pickle

def main(algID   =  'tabQL_Cest_vi_t2',
         feedback_type = 'ranked-feedback',
         simInfo = '_50trials_10fb_ranked',
         env_size = 'small',
         trial_count = 1,
         episode_count = 1000,
         max_steps = 500,
         L  = np.array([10,10,10,10]),
         C  = np.array([0.9,0.8,0.6,0.3]),
         prior_alpha  = 1.0,
         prior_beta   = 1.0,
         no_reward = False,
         C_fixed = None,
         update_Cest_interval = 5,
         active_feedback_type = 'count',
         reset_env_random = False,
         randomise_trajectory = False,
         oracle4fb_type = 'old',
         oracle4al_type = 'old',
         ):

    print(f"start--{algID} {simInfo}")
    
    monitor = RLmon(['return', 'Cest', 'alpha', 'beta'])
    all_feedback = []
    
    for k in range(trial_count):
        print('trial: {0}'.format(k))
        
        trainer = PacmanTrainer(
            algID=algID,
            env_size=env_size,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            C_fixed=C_fixed,
            no_reward=no_reward,
            active_feedback_type=active_feedback_type
        )
        
        # Setup ORACLE
        oracle_h = agent('tabQLgreedy', trainer.env.nStates(), len(trainer.env.action_list()))
        if env_size == 'small':
            if oracle4fb_type == 'new': 
                oracle_h.load('learnedStates/pacman_small_tabQL_oracle_randomised_env.pkl')
            else:
                oracle_h.load('learnedStates/_nochase_smalltabQLgreedy.pkl')
        elif env_size == 'medium':
            oracle_h.load('/Users/jonathanerskine/University of Bristol/RL-multiple-human-feedback/learnedStates/_randenv_mediumtabQLgreedy.pkl')
        elif env_size == 'medium_sparse':
            oracle_h.load('/Users/jonathanerskine/University of Bristol/RL-multiple-human-feedback/learnedStates/_randenv_medium_sparsetabQLgreedy.pkl')
        
        oracle_h.alpha = 0
        oracle_h_act = oracle_h
        
        fbs = []
        totalRW_list = []
        
        for i in range(episode_count):
            trainer.reset_episode(random=reset_env_random)
            
            for j in range(max_steps):
                # Simulated 'human' feedback generation (by using ORACLE)
                fb = [[] for _ in range(len(C))]
                if np.any(L > 0.0):
                    fb = generate_feedback(
                            trajectory=trainer.trajectory,
                            C=C,
                            L=L,
                            action_list=trainer.env.action_list(), 
                            agent_h=trainer.agent, 
                            oracle_h=oracle_h,
                            oracle_h_act=oracle_h_act, 
                            end_of_episode=False, 
                            no_reward=no_reward, 
                            feedback_type=feedback_type, 
                            active_feedback_type=active_feedback_type) #None if active_feedback_type is None else 'none') # Oracle gives feedback every step or based on simple logic here

                # Perform step
                # Note: In the original main_oracle, it calls agent.act twice, once to get action and once to update at end.
                # Here we use trainer.step which encapsulates the agent call.
                
                # To perfectly match main_oracle's feedback timing, we might need more control.
                # But trainer.step is the "performant" way we want to promote.
                
                action_idx, ob, rw, done = trainer.step(feedback=fb, update_Cest=False)

                if done or j == max_steps - 1:
                    # Final update with active learning feedback at end of episode
                    if active_feedback_type is not None and np.any(L > 0.0):
                        fb = generate_feedback(
                                trajectory=trainer.trajectory,
                                C=C,
                                L=L,
                                action_list=trainer.env.action_list(), 
                                agent_h=trainer.agent, 
                                oracle_h=oracle_h,
                                oracle_h_act=oracle_h_act, 
                                end_of_episode=True, 
                                no_reward=no_reward, 
                                feedback_type=feedback_type, 
                                active_feedback_type=active_feedback_type)
                    
                    update_Cest = ((i+1) % update_Cest_interval == 0)
                    trainer.agent.act(action_idx, ob, rw, done, fb, C, update_Cest=update_Cest)
                    trainer.agent.prev_obs = None
                    fbs.append(fb)
                    break

            totalRW_list.append(trainer.totRW)
            if i % 20 == 0:
                print(f"{k}, {i}: Ce: {trainer.agent.Ce} \t total reward: mean:{np.mean(totalRW_list):+.1f}")
                totalRW_list = []
            
            monitor.store('return', trainer.totRW)
            monitor.store('Cest', trainer.agent.Ce)
            if hasattr(trainer.agent, 'sum_of_right_feedback'):
                monitor.store('alpha', trainer.agent.sum_of_right_feedback + trainer.agent.a)
                monitor.store('beta',  trainer.agent.sum_of_wrong_feedback + trainer.agent.b)
        
        all_feedback.append(fbs)
        monitor.store(done=True)

    fname = 'results/results_' + 'Env_' + str(env_size) + '_' + str(algID) + str(simInfo) + str(active_feedback_type)
    monitor.saveData(fname)

    with open(f'results/'+'feedback_Env_' + str(env_size) + '_' + str(algID) + str(simInfo) + str(active_feedback_type)+'.pkl','wb') as f:
        pickle.dump(all_feedback,f)

if __name__ == '__main__':
    main()
