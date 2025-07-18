import random
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    test_env = gym.make('LunarLander-v3')
    test_env.close()
    ENV_NAME = 'LunarLander-v3'
    SOLVED_THRESHOLD = 200
    print("Using LunarLander-v2 environment")
except:
    ENV_NAME = 'CartPole-v1'
    SOLVED_THRESHOLD = 195
    print("LunarLander not available, using CartPole-v1 environment")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.memory = ReplayBuffer(10000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        self.learn_step = 0
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=1000,  render=False):
    env = gym.make(ENV_NAME, render_mode='human' if render else None)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    scores = []
    solved_episodes = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            if render:
                env.render()
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.replay()
        scores.append(total_reward)
        
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            if avg_score >= SOLVED_THRESHOLD and len(solved_episodes) == 0:
                solved_episodes.append(episode)
                print(f"Environment solved in {episode} episodes! Average score: {avg_score:.2f}")
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    return agent, scores, solved_episodes

def plot_results(scores, solved_episodes):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    if len(scores) >= 100:
        moving_avg = [np.mean(scores[i:i+100]) for i in range(len(scores)-99)]
        plt.plot(range(99, len(scores)), moving_avg, 'r-', linewidth=2, label='100-episode average')
        plt.axhline(y=SOLVED_THRESHOLD, color='g', linestyle='--', label='Solved threshold')
        plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=50, alpha=0.7)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.axvline(x=SOLVED_THRESHOLD, color='g', linestyle='--', label='Solved threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def test_agent(agent, episodes=10, render=False):
    if render:
        env = gym.make(ENV_NAME, render_mode='human')
    else:
        env = gym.make(ENV_NAME)
    
    test_scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.q_network(state_tensor).argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        test_scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward:.2f}")
    
    env.close()
    
    avg_score = np.mean(test_scores)
    print(f"\nTest Results:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Standard Deviation: {np.std(test_scores):.2f}")
    print(f"Max Score: {max(test_scores):.2f}")
    print(f"Min Score: {min(test_scores):.2f}")
    
    return test_scores

if __name__ == "__main__":
    print("Starting Lunar Lander RL Training...")
    print("=" * 50)
    agent, scores, solved_episodes = train_agent(episodes=1000, render=True)
    
    plot_results(scores, solved_episodes)
    print("\nTesting trained agent...")
    test_scores = test_agent(agent, episodes=5, render=True)

    torch.save(agent.q_network.state_dict(), 'lunar_lander_dqn.pth')
    print("\nModel saved as 'lunar_lander_dqn.pth'")
    
    print("\nTraining complete!")
    if ENV_NAME == 'LunarLander-v2':
        print("The agent should learn to land the lunar lander successfully.")
        print("Scores above 200 indicate successful landing.")
    else:
        print("The agent learned to balance the CartPole.")
        print("Scores above 195 indicate successful balancing.")

def visualize_environment():
    """Create and display the environment"""
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    state, _ = env.reset()
    
    img = env.render()
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f'{ENV_NAME} Environment')
    plt.axis('off')
    plt.show()
    
    env.close()
    
    print("Environment Information:")
    print(f"Environment: {ENV_NAME}")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    if ENV_NAME == 'LunarLander-v2':
        print("\nActions:")
        print("0: Do nothing")
        print("1: Fire left orientation engine")
        print("2: Fire main engine")
        print("3: Fire right orientation engine")
        
        print("\nState variables:")
        print("0: Horizontal position")
        print("1: Vertical position")
        print("2: Horizontal velocity")
        print("3: Vertical velocity")
        print("4: Angle")
        print("5: Angular velocity")
        print("6: Left leg contact")
        print("7: Right leg contact")
    else:
        print("\nActions:")
        print("0: Push cart to the left")
        print("1: Push cart to the right")
        
        print("\nState variables:")
        print("0: Cart position")
        print("1: Cart velocity")
        print("2: Pole angle")
        print("3: Pole angular velocity")

visualize_environment()