import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class Policy(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        
        # Detect environment and action space
        env = gym.make('CarRacing-v2')
        self.continuous = isinstance(env.action_space, gym.spaces.Box)
        
        # Determine state and action dimensions
        self.state_dim = np.prod(env.observation_space.shape)
        self.action_dim = env.action_space.shape[0] if self.continuous else env.action_space.n
        env.close()

        # Network Architecture
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Policy head
        if self.continuous:
            self.mean_layer = nn.Linear(64, self.action_dim)
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        else:
            self.action_head = nn.Linear(64, self.action_dim)

        # Move to device
        self.to(device)

        # Optimization setup
        self.optimizer = optim.Adam(self.parameters(), lr=0.0015)
        
        # Training parameters
        self.clip_ratio = 0.2
        self.gamma = 0.98
        self.lmbda = 0.95
        self.entropy_coef = 0.005

        # Training state
        self.training_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'dones': []
        }

    def forward(self, x):
        x = self.network(x)
        
        if self.continuous:
            mean = self.mean_layer(x)
            std = torch.exp(self.log_std)
            return mean, std
        else:
            return F.softmax(self.action_head(x), dim=-1)

    def act(self, state):
        state = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)
        
        if self.continuous:
            mean, std = self(state)
            action_dist = torch.distributions.Normal(mean, std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            action = torch.clamp(action, -1, 1)  # Clip for racing car env
            return action.detach().cpu().numpy()[0]
        else:
            probs = self(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.detach().cpu().item()

    def train(self, num_episodes=50, max_steps=50):
        """Training loop compatible with main script's expectations"""
        env = gym.make('CarRacing-v2', continuous=self.continuous)

        for episode in range(num_episodes):
            state, _ = env.reset()
            state = state.flatten()
            done = False
            total_reward = 0

            for step in range(max_steps):
                # Sample action
                action = self.act(state)
                
                # Interact with environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.training_buffer['states'].append(state)
                self.training_buffer['actions'].append(action)
                self.training_buffer['rewards'].append(reward)
                self.training_buffer['dones'].append(done)
                
                state = next_state.flatten()
                total_reward += reward

                if done:
                    break

            # Periodic policy update (every episode or few episodes)
            if len(self.training_buffer['states']) >= max_steps:
                self._update_policy()
                self._reset_training_buffer()

            print(f"Episode {episode}, Total Reward: {total_reward}")

        env.close()

    def _update_policy(self):
        """Internal method to update policy using PPO"""
        # Convert buffer to tensors
        states = torch.FloatTensor(self.training_buffer['states']).to(self.device)
        actions = torch.FloatTensor(self.training_buffer['actions']).to(self.device)
        rewards = np.array(self.training_buffer['rewards'])
        dones = np.array(self.training_buffer['dones'])

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = np.array(returns)

        # Compute advantages (simple version)
        advantages = returns - returns.mean()

        # PPO update
        for _ in range(3):  # Multiple epochs
            if self.continuous:
                mean, std = self(states)
                action_dist = torch.distributions.Normal(mean, std)
                log_probs = action_dist.log_prob(actions).sum(dim=-1)
            else:
                probs = self(states)
                action_dist = torch.distributions.Categorical(probs)
                log_probs = action_dist.log_prob(actions)

            # Compute policy loss
            ratio = torch.exp(log_probs)
            surr1 = ratio * torch.FloatTensor(advantages).to(self.device)
            surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * torch.FloatTensor(advantages).to(self.device)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Optimize
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

    def _reset_training_buffer(self):
        """Reset training buffer after update"""
        self.training_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'dones': []
        }

    def save(self, filename='model.pt'):
        torch.save(self.state_dict(), filename)

    def load(self, filename='model.pt'):
        self.load_state_dict(torch.load(filename, map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret