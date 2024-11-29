import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity   # buffer capacity
        self.alpha = alpha         # alpha value for prioritized experience replay, control hm prioritization
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    #this is the function which storage the expereince in the buffer
    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        #control if the buffer has reached the maximum capacity
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
         
        #New experience get maximum priority
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Empty buffer")
        
        n_samples = min(batch_size, len(self.buffer))
        
        # Calculate sampling probabilities based on priorities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), n_samples, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = np.stack(states).astype(np.float32) / 255.0
        next_states = np.stack(next_states).astype(np.float32) / 255.0
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones, indices, weights)

    #update function for the priorities
    def update_priorities(self, indices, priorities):
        priorities = priorities.flatten()
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority) + 1e-6

# NN model for the DDQN
class DDQNNet(nn.Module):
    def __init__(self, input_shape=(3, 96, 96), num_actions=3):
        super(DDQNNet, self).__init__()
        
        # Deeper network with batch normalization
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 128
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        # Better initialization
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.414)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Policy:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        
        self.q_network = DDQNNet().to(self.device)
        self.target_network = DDQNNet().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Basic hyperparameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4)
        self.buffer = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
        self.batch_size = 64
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0   #exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Adjusted for more episodes
        self.beta = 0.4
        self.beta_increment = 0.001
        self.target_update_freq = 1000   #how often to update the target network
        self.continuous = True
        self.steps = 0

    def process_reward(self, reward, action, done):
        # Keep reward processing simple
        processed_reward = reward

        # Extract action components
        steering = action[0]
        gas = action[1]
        brake = action[2]
        
        
        # Basic penalty for being stationary too long
        if self.steps_stopped > 20:
            processed_reward = -10
            done = True

        # Additional penalty for staying at extreme steering angles
        if abs(steering) > 0.6:
            processed_reward -= 0.5

            # Encourage straight driving when possible
        if abs(steering) < 0.1:
            processed_reward += 0.1

            # Encourage speed but with more nuanced controls
        if gas > 0.1:
            # Reward higher speeds only when steering is reasonable
            speed_bonus = gas * (1.0 - abs(steering))
            processed_reward += speed_bonus
            
        return processed_reward, done

    def act(self, state, training=False):
        if training and random.random() < self.epsilon:
            # Simple random actions
            return np.array([
                np.random.uniform(-0.8, 0.8),    # steering
                np.random.uniform(0, 1),     # gas
                np.random.uniform(0, 0.5)      # brake
            ])
        
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device) / 255.0
            q_values = self.q_network(state)
            actions = torch.tanh(q_values).cpu().numpy()[0]
        
        return actions
    
    def train_step(self):
        if len(self.buffer.buffer) < self.batch_size:
            return
    
        # Sample the batch based on priorities
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size, beta=self.beta)
    
        # Move to GPU in one batch
        states = torch.from_numpy(states).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.from_numpy(next_states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)

        # Double Q-learning target calculation
        with torch.no_grad():

            # Get actions from main network
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.argmax(dim=1)
        
            # Get Q-values from target network for next states
            next_target_q_values = self.target_network(next_states)
            next_target_q_values = next_target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
            # Calculate target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_target_q_values

        # Current Q-values
        current_q_values = self.q_network(states)
        action_indices = torch.argmax(actions, dim=1)
        predicted_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Calculate loss with importance sampling weights
        td_errors = torch.abs(predicted_q_values - target_q_values)
        loss = (weights * F.smooth_l1_loss(predicted_q_values, target_q_values.detach(), reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in buffer
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Update beta after each training step
        self.beta = min(1.0, self.beta + self.beta_increment)

    def train(self, num_episodes=200):  # Increased number of episodes
        env = gym.make('CarRacing-v2', continuous=self.continuous)
        best_reward = -float('inf')
        recent_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            episode_steps = 0
            self.steps_stopped = 0
            
            while not done and episode_steps < 1000:

                #this 4 line is the state collection step
                action = self.act(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.buffer.add(state.copy(), action, reward, next_state.copy(), done)
                

                if len(self.buffer.buffer) >= self.batch_size:
                    self.train_step()
                
                state = next_state.copy()
                total_reward += reward
                episode_steps += 1
                self.steps += 1
                
                # Update epsilon after each step
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
            
            recent_rewards.append(total_reward)
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)
            
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}, "
                  f"Moving Avg: {np.mean(recent_rewards):.2f}")
            
            if total_reward > best_reward:
                best_reward = total_reward
                self.save('best_policy_car_racing.pt')

    def save(self, filename='policy_car_racing.pt'):
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, filename)

    def load(self, filename='policy_car_racing.pt'):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)