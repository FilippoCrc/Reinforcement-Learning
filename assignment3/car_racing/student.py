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
        self.alpha = alpha    # alpha value for prioritized experience replay, control how much prioritization
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

# sampling function for the buffer
    def sample(self, batch_size, beta=0.4):
        
        n_samples = min(batch_size, len(self.buffer))

        
        #rank based sampling

        # Get priorities of all experiences in buffer
        priorities = self.priorities[:len(self.buffer)]
        
        # Get ranks (ascending order, so highest priority = rank 1)
        ranks = len(priorities) - np.argsort(np.argsort(priorities))
        
        # Calculate probabilities based on rank
        # P(i) = 1 / (rank(i))^α
        probs = (1 / ranks) ** self.alpha
        probs /= probs.sum()
        
        """
        # Random sampling
        # Converts priorities to probabilities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum() 
        
        """
        
        # Sample experiences indices based on probs (before computed)
        indices = np.random.choice(len(self.buffer), n_samples, p=probs)

        # Calculate importance sampling weights
        # reduces the bias introduced by the non-uniform probabilities
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Normalize the experience
        states = np.stack(states).astype(np.float32) / 255.0
        next_states = np.stack(next_states).astype(np.float32) / 255.0
        
        actions = np.array(actions, dtype=np.int64)  
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones, indices, weights)

    #After the training_step, this function updates the priorities based on the new TD errors
    def update_priorities(self, indices, priorities):
        priorities = priorities.flatten()
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority) + 1e-6

# NN model for the DDQN
class DDQNNet(nn.Module):

    def __init__(self, input_shape=(3, 96, 96), num_actions=5):  # Changed from 3 to 5 actions
        super(DDQNNet, self).__init__()
        
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
        self.fc2 = nn.Linear(512, num_actions)  # Outputs Q-values for each discrete action
        
        #this inizialize the weight and the bias
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
        
        # CHANGE: Changed to discrete actions
        self.continuous = False  # Now using discrete actions
        self.num_actions = 5    # Define number of discrete actions
        
        # Action mappings for discrete action space basic gym
        # 0: Do nothing
        # 1: Steer left
        # 2: Steer right
        # 3: Gas
        # 4: Brake

        self.q_network = DDQNNet(num_actions=self.num_actions).to(self.device)
        self.target_network = DDQNNet(num_actions=self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        #initialize the optimizer and the buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4)
        self.buffer = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
        self.batch_size = 32
        
        # Basic hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.beta = 0.4
        self.beta_increment = 0.001
        self.target_update_freq = 1000
        self.steps = 0

    def act(self, state, training=False):
        #Epsilon greedy policy
        if training and random.random() < self.epsilon:
            # Random discrete action
            action_idx = random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device) / 255.0
                q_values = self.q_network(state)
                action_idx = q_values.argmax(dim=1).item()
        
        return action_idx  # Return discrete index for discrate actions

    """
    def process_reward(self, reward, action_idx, done):
        # Keep reward processing simple
        processed_reward = reward

        # Extract action components
        steering_left = action_idx[0]
        steering_right = action_idx[1]
        gas = action_idx[2]
        brake = action_idx[3]
        
        
        # Basic penalty for being stationary too long
        if self.steps_stopped > 20:
            processed_reward = -10
            done = True

        # Additional penalty for staying at extreme steering angles
        if (abs(steering_left) > 0.6 | abs(steering_right) > 0.6):
            processed_reward -= 0.5


            # Encourage speed but with more nuanced controls
        if gas > 0.1:
            # Reward higher speeds only when steering is reasonable
            speed_bonus = gas * 0.5
            processed_reward += speed_bonus
            
        return processed_reward, done
        """

    def train_step(self):
        if len(self.buffer.buffer) < self.batch_size:
            return
        # Samples experiences from the prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size, beta=self.beta)
    
        # Move to GPU
        states = torch.from_numpy(states).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.from_numpy(next_states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)

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
        predicted_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate loss with importance sampling weights
        td_errors = torch.abs(predicted_q_values - target_q_values)
        loss = (weights * F.smooth_l1_loss(predicted_q_values, target_q_values.detach(), reduction='none')).mean()

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

    def train(self, num_episodes=1500):
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
                action = self.act(state, training=True)  # Get both action types
                next_state, reward, terminated, truncated, _ = env.step(action)  # Use continuous action for environment
                done = terminated or truncated
                # Store discrete action in buffer
                self.buffer.add(state.copy(),action, reward, next_state.copy(), done)
                
                """
                Uncomment for Use the Processed Reward instead of the original reward
                
                # Process the reward before adding to buffer
                processed_reward  = self.process_reward(reward, action, done, state)
                self.buffer.add(state.copy(), action, processed_reward, next_state.copy(), done)

                """

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
                self.save('model.pt')

    def save(self, filename='model.pt'):
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, filename)

    def load(self, filename='model.pt'):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret