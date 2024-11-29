import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Empty buffer")
        
        n_samples = min(batch_size, len(self.buffer))
        
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), n_samples, p=probs)
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

    def update_priorities(self, indices, priorities):
        priorities = priorities.flatten()
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority) + 1e-6

class DDQNNet(nn.Module):
    def __init__(self, input_shape=(3, 96, 96), num_actions=3):
        super(DDQNNet, self).__init__()
        
        # Deeper network for better feature extraction
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
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
        
        # Adjusted hyperparameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4)
        self.buffer = PrioritizedReplayBuffer(capacity=100000)
        self.batch_size = 128
        
        # Modified parameters for better exploration/exploitation
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999  # Slower decay
        self.beta = 0.4
        self.beta_increment = 0.001
        self.target_update_freq = 1000
        self.continuous = True
        self.steps = 0
        
        # Track velocities for reward shaping
          # Track velocities and rewards
        self.prev_velocity = 0
        self.steps_stopped = 0
        self.reward_history = []  # Added this line
        self.moving_avg_size = 100  # Added this line

    def process_reward(self, reward, action, done):
        # Base reward from environment
        processed_reward = reward
        
        # Extract action components
        steering = action[0]
        gas = action[1]
        brake = action[2]
        
         # Encourage speed
        if gas > 0.2:
            speed_bonus = gas * 0.1
            processed_reward += speed_bonus
        
        # Penalize unnecessary braking
        if brake > 0.1 and gas > 0.3:
            processed_reward -= 0.3
    
        # Penalize extreme steering
        steering_penalty = abs(steering) * 0.1
        processed_reward -= steering_penalty
    
        # Encourage straight driving when possible
        if abs(steering) < 0.1:
            processed_reward += 0.1
    
        # Early termination conditions
        if self.steps_stopped > 30:  # Reduced from 50 to be more aggressive
            processed_reward = -50
            done = True

        processed_reward = np.clip(processed_reward, -10, 10)
            
        return processed_reward, done

    def act(self, state, training=False):
        if training and random.random() < self.epsilon:
            # Smarter exploration strategy
            if random.random() < 0.8:  # 70% of exploration actions
                # Generate actions that encourage movement
                steering = np.random.uniform(-0.3, 0.3)  # Less extreme steering
                gas = np.random.uniform(0.4, 0.8)       # Ensure forward movement
                brake = np.random.uniform(0, 0.2)       # Minimal braking
                return np.array([steering, gas, brake])
            else:
                return np.array([
                np.random.uniform(-1, 1),    # steering
                np.random.uniform(0.3, 1.0),  # gas
                np.random.uniform(0, 0.3)     # brake
            ])
    
        
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device) / 255.0
            q_values = self.q_network(state)
            actions = torch.tanh(q_values).cpu().numpy()[0]
        
        # Post-process actions
        actions[0] = np.clip(actions[0], -0.8, 0.8)  # Limit steering angle
        actions[1] = np.clip(actions[1], 0.3, 1.0)   # Maintain higher gas
        actions[2] = np.clip(actions[2], 0, 0.3)    # Limit maximum braking

        # Prevent simultaneous high gas and brake
        if actions[1] > 0.5:  # If gas is high
            actions[2] = 0.0   # No brake
        
        return actions
    
    def train_step(self):
        if len(self.buffer.buffer) < self.batch_size:
            return
    
        # Sample and prepare batch
        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.batch_size, beta=self.beta)
    
        # Move to GPU in one batch
        states = torch.from_numpy(states).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.from_numpy(next_states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)

        # Double Q-learning target
        with torch.no_grad():
            # Get actions from main network
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.argmax(dim=1)
        
            # Get Q-values from target network
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

        # Update priorities
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.beta = min(1.0, self.beta + self.beta_increment)

    def train(self, num_episodes=50):
        env = gym.make('CarRacing-v2', continuous=self.continuous)
        best_reward = -float('inf')
        recent_rewards = []
        
        state_shape = (96, 96, 3)
        state_buffer = np.zeros(state_shape, dtype=np.float32)
        next_state_buffer = np.zeros(state_shape, dtype=np.float32)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            np.copyto(state_buffer, state)
            done = False
            total_reward = 0
            episode_steps = 0
            self.steps_stopped = 0
            
            while not done and episode_steps < 1000:
                action = self.act(state_buffer, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                np.copyto(next_state_buffer, next_state)
                
                # Process reward and check for early termination
                processed_reward, early_done = self.process_reward(reward, action, terminated or truncated)
                done = terminated or truncated or early_done
                
                self.buffer.add(state_buffer.copy(), action, processed_reward, next_state_buffer.copy(), done)
                
                if len(self.buffer.buffer) >= self.batch_size:
                    self.train_step()
                
                np.copyto(state_buffer, next_state_buffer)
                total_reward += reward  # Track original reward for monitoring
                episode_steps += 1
                self.steps += 1
                
                # Adaptive epsilon decay
                if len(recent_rewards) > 0 and np.mean(recent_rewards) > 0:
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
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
            'reward_history': self.reward_history
        }, filename)

    def load(self, filename='policy_car_racing.pt'):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)
        self.reward_history = checkpoint.get('reward_history', [])