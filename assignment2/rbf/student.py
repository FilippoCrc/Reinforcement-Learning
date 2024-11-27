import random
import numpy as np
# import gymnasium as gym
# import time
# from gymnasium import spaces
# import os
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import pickle


class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

""" class RBFFeatureEncoder:
    def __init__(self, env):
         # TODO init rbf encoder
        self.env = env
        self.n_components = 100 

        #  sample observation from the envirenment
        self.total_observation = np.array([env.observation_space.sample() for x in range(20000)])

        #inizialization of the scaler and encoder
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.encoder = RBFSampler(gamma=0.5, n_components=100)

        #learn the scaling parameters
        self.scaler.fit(self.total_observation)
        #scale the data
        scaled_data = self.scaler.transform(self.total_observation)
        #fit the encoder
        self.encoder.fit(scaled_data)

    def encode(self, state): # modify
        # TODO use the rbf encoder to return the features
        features = self.scaler.transform([state])
        features = self.encoder.transform(features)
        return features.reshape(-1)

    @property
    def size(self): # modify
        # TODO return the number of features
        return self.n_components """
    
class RBFFeatureEncoder:
    def __init__(self, env, n_components=20, gamma=1):
        self.env = env
        self.n_components = n_components # numbers of features
        self.gamma = gamma

        # Collect samples from the environment
        total_observation = np.array([env.observation_space.sample() for _ in range(20000)])

        # Compute mean and standard deviation for scaling
        self.mean = np.mean(total_observation, axis=0)
        self.std = np.std(total_observation, axis=0)

        # Generate random projection weights and biases for RBF
        self.random_weights = np.sqrt(2 * self.gamma) * np.random.normal(size=(env.observation_space.shape[0], n_components))
        self.random_biases = np.random.uniform(0, 2 * np.pi, n_components)
  

    def encode(self, state):
        # Scale the state
        scaled_state = (state - self.mean) / self.std
        # Apply the RBF transformation
        rbf_features = np.cos(np.dot(scaled_state, self.random_weights) + self.random_biases)
        return rbf_features

    @property
    def size(self):
        # Number of features
        return self.n_components

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder, alpha=0.01, alpha_decay=1, 
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.2, lambda_=0.9): # modify if you want (e.g. for forward view)
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_
        
    def Q(self, feats):
        feats = feats.reshape(-1,1)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)
        # TODO update the weights

        #current Q state
        Current_Q = self.Q(s_feats)[action]

        # check if we are in the terminal state
        if done:
            Next_Q = 0  
        else:
            Next_Q = max(self.Q(s_prime_feats))

        #error computation
        error = reward + self.gamma * Next_Q - Current_Q

        # Update eligibility trace 
        self.traces *= self.gamma* self.lambda_ 
        self.traces[action] += s_feats

        # Update weights
        self.weights[action] += self.alpha * error * self.traces[action]
        
    def update_alpha_epsilon(self): # do not touch
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None): # do not touch
        if epsilon is None: epsilon = self.epsilon
        if random.random()<epsilon:
            return self.env.action_space.sample()
        return self.policy(state)
       
        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        print(f'ep | eval | epsilon | alpha')
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)
                
                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
