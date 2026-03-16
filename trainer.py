import numpy as np
import envPacMan as environment
from agent import agent
from feedback import Trajectory, get_active_utility
import pickle

class PacmanTrainer:
    def __init__(self, 
                 algID='tabQL_Cest_vi_t2', 
                 env_size='small', 
                 prior_alpha=1.0, 
                 prior_beta=1.0, 
                 C_fixed=None,
                 no_reward=False,
                 active_feedback_type='count',
                 active_learning_threshold=3.0):
        
        self.algID = algID
        self.env_size = env_size
        self.no_reward = no_reward
        self.active_feedback_type = active_feedback_type
        self.active_learning_threshold = active_learning_threshold
        
        self.env = environment.env(env_size)
        self.agent = agent(algID, self.env.nStates(), len(self.env.action_list()), 
                           a=prior_alpha, b=prior_beta, 
                           C_fixed=C_fixed)
        
        self.action_list = self.env.action_list()
        self.trajectory = Trajectory()
        
        self.reset_episode()

    def reset_episode(self, random=True, pellet_random=False):
        self.env.reset(random=random, pellet_random=pellet_random)
        self.agent.prev_obs = None
        self.ob = self.env.st2ob()
        self.rw = 0
        self.done = False
        self.totRW = 0
        self.step_count = 0
        self.trajectory.reset()

    def step(self, feedback=None, update_Cest=False):
        """
        Perform a single step in the environment.
        feedback: list of feedback objects if provided by human or oracle.
        """
        if feedback is None:
            feedback = [[]]
            
        # call agent to get action
        # Note: In the interactive case, the action might have already been 'taken' to show the next frame.
        # But for consistency with main_oracle, we follow the act -> step cycle.
        action_idx = self.agent.act(0, self.ob, self.rw, self.done, feedback, 0.5, update_Cest=update_Cest)
        
        # store for trajectory
        prev_ob = self.ob
        
        # call environment
        self.ob, self.rw, self.done = self.env.step(self.action_list[action_idx])
        self.totRW += self.rw
        self.step_count += 1
        
        # record trajectory
        self.trajectory.append(state=prev_ob, action=action_idx, reward=self.rw, done=self.done, optimal_action=None)
        
        return action_idx, self.ob, self.rw, self.done

    def wants_feedback(self, obs, action):
        """
        Check if the current state-action pair requires feedback based on active learning.
        """
        if self.active_feedback_type is None:
            return True
            
        utility = get_active_utility(obs, action, self.agent, mode=self.active_feedback_type, no_reward=self.no_reward)
        
        # For 'count', utility is 1/(Nsa + certainty). 
        # Threshold in flask_app was (nsa + certainty) < 3, which is utility > 1/3.
        # So we compare utility with 1/threshold or just define threshold on utility.
        
        if self.active_feedback_type == 'count':
            # Align with previous flask_app logic: Nsa + certainty < threshold
            # utility = 1 / (Nsa + certainty) => (Nsa + certainty) = 1/utility
            return (1.0 / utility) < self.active_learning_threshold
        elif self.active_feedback_type == 'entropy':
            # For entropy, higher means more uncertain.
            return utility > self.active_learning_threshold
            
        return True

    def run_episode(self, max_steps=500, random_start=True):
        """Run a full episode internally and store the trajectory."""
        self.reset_episode(random=random_start)
        for _ in range(max_steps):
            # Step with empty feedback
            _, _, _, done = self.step(feedback=[[]], update_Cest=False)
            if done:
                break
        return self.trajectory

    def select_feedback_indices(self, num_feedbacks=10):
        """
        Identify the most informative indices from the current trajectory
        using the same logic as feedback.py / main_oracle.py.
        """
        if len(self.trajectory) == 0:
            return []
            
        U = np.zeros(len(self.trajectory))
        for n, (s, a) in enumerate(zip(self.trajectory.state, self.trajectory.action)):
            U[n] = get_active_utility(s, a, self.agent, mode=self.active_feedback_type, no_reward=self.no_reward)
        
        # Use the logic from get_index_for_feedback in feedback.py
        # Normalise utility
        if np.std(U) > 0:
            U = (U - np.mean(U)) / np.std(U)
            
        N_fb = min(len(U), int(num_feedbacks))
        
        if N_fb == len(U):
            idx = np.arange(len(U))
        else:
            # Pick top N indices (with small noise for tie-breaking/variety)
            idx = np.argpartition(-U + np.random.randn(len(U)) * 0.1, N_fb)[:N_fb]
        return sorted(idx.tolist())

    def get_brain(self):
        """Return the learnable parts of the agent for persistence."""
        return {
            'Q': self.agent.Q,
            'hp': self.agent.hp,
            'hm': self.agent.hm,
            'Ce': self.agent.Ce,
            'sum_of_right_feedback': getattr(self.agent, 'sum_of_right_feedback', 0),
            'sum_of_wrong_feedback': getattr(self.agent, 'sum_of_wrong_feedback', 0),
            'Nsa': getattr(self.agent, 'Nsa', np.zeros_like(self.agent.Q))
        }

    def load_brain(self, brain_data):
        """Load brain data into the current agent."""
        if brain_data:
            from scipy.special import psi as scipy_psi

            self.agent.Q  = brain_data['Q']
            self.agent.hp = brain_data['hp']
            self.agent.hm = brain_data['hm']
            self.agent.Ce = brain_data['Ce']

            # nTrainer is not persisted separately — derive it from Ce
            self.agent.nTrainer = len(brain_data['Ce'])

            self.agent.sum_of_right_feedback = brain_data.get(
                'sum_of_right_feedback', np.zeros(self.agent.nTrainer))
            self.agent.sum_of_wrong_feedback = brain_data.get(
                'sum_of_wrong_feedback', np.zeros(self.agent.nTrainer))

            # Ensure a/b are arrays of length nTrainer (init block normally does this)
            if not hasattr(self.agent.a, "__len__"):
                self.agent.a = np.ones(self.agent.nTrainer) * self.agent.a
            if not hasattr(self.agent.b, "__len__"):
                self.agent.b = np.ones(self.agent.nTrainer) * self.agent.b

            # Recompute psi values that tabQL_Cest_vi needs on every call
            self.agent.psi_for_hr = (
                scipy_psi(self.agent.sum_of_right_feedback + self.agent.a) -
                scipy_psi(self.agent.sum_of_right_feedback +
                          self.agent.sum_of_wrong_feedback +
                          self.agent.a + self.agent.b)
            )
            self.agent.psi_for_hw = (
                scipy_psi(self.agent.sum_of_wrong_feedback + self.agent.b) -
                scipy_psi(self.agent.sum_of_right_feedback +
                          self.agent.sum_of_wrong_feedback +
                          self.agent.a + self.agent.b)
            )

            self.agent.Nsa = brain_data.get(
                'Nsa', np.zeros_like(self.agent.Q))

    def save_agent(self, fname):
        self.agent.save(fname)

    def load_agent(self, fname):
        self.agent.load(fname)
