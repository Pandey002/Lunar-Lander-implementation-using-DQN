🌕 Lunar Lander Implementation using Deep Q-Network (DQN)

This project implements a Deep Q-Network (DQN) agent to solve the OpenAI Gym / Gymnasium LunarLander environment.
The agent learns optimal landing behavior using experience replay, target networks, and epsilon-greedy exploration, achieving stable learning in a continuous state space with discrete actions.

If the LunarLander environment is unavailable, the code automatically falls back to CartPole-v1.

🚀 Features

✅ Deep Q-Network (DQN) using PyTorch

🧠 Experience Replay Buffer

🎯 Target Network for stable learning

🔄 Epsilon-Greedy Exploration with decay

📈 Training score visualization

🎮 Real-time environment rendering


| Technique            | Purpose                             |
| -------------------- | ----------------------------------- |
| Experience Replay    | Breaks correlation between samples  |
| Target Network       | Stabilizes Q-value updates          |
| Epsilon Decay        | Balances exploration & exploitation |
| Fixed Target Updates | Prevents oscillations               |





⚙️ Environment Details
LunarLander

State Space: 8 continuous variables

Action Space: 4 discrete actions

Solved Threshold: ≥ 200 average reward over 100 episodes




Actions:
0 → Do nothing
1 → Fire left orientation engine
2 → Fire main engine
3 → Fire right orientation engine



State Variables:
[ x_position,
  y_position,
  x_velocity,
  y_velocity,
  angle,
  angular_velocity,
  left_leg_contact,
  right_leg_contact ]



🧪 Training Process

Replay buffer size: 10,000

Batch size: 64

Target update frequency: 100 steps

Initial epsilon: 1.0

Minimum epsilon: 0.01

Epsilon decay: 0.995

Episodes: 1000

The environment is considered solved once the average reward over 100 episodes exceeds the threshold.

📊 Visualization

The training process includes:

📈 Episode reward curve

📉 Moving average (100 episodes)

📊 Score distribution histogram

🎮 Optional real-time rendering

▶️ How to Run
1️⃣ Install Dependencies
pip install gymnasium[box2d] torch numpy matplotlib


2️⃣ Run Training
python main.py


The script will:

Train the agent

Render the environment (optional)

Plot training statistics

Test the trained agent


🧪 Testing the Agent

After training, the agent is evaluated over multiple test episodes using a greedy policy (no exploration).

Metrics reported:

Average score

Maximum & minimum score

Standard deviation

💾 Model Saving

The trained DQN model is saved using:

torch.save(agent.q_network.state_dict(), 'lunar_lander_dqn.pth')

🧠 Key Learning Outcomes

Understanding DQN architecture and training

Handling continuous state spaces with neural networks

Stabilizing reinforcement learning using target networks

Practical experience with Gymnasium environments

RL training visualization and evaluation

🔮 Future Improvements

Double DQN

Dueling DQN Architecture

Prioritized Experience Replay

Hyperparameter tuning

Training speed optimization with vectorized environments

📌 Technologies Used

Python

PyTorch

Gymnasium (OpenAI Gym)

NumPy

Matplotlib

👤 Author:
Hitesh Kumar
AI / ML Enthusiast | Reinforcement Learning Practitioner
