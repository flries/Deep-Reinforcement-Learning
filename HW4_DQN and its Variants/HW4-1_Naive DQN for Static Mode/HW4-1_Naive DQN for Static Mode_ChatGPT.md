# HW4-1_Naive DQN for Static Mode
## Prompt:
Explain Naive DQN (Deep Q-Network).

## Response:
**Naive Deep Q-Network (Naive DQN)** is an early and simplified form of using deep learning for reinforcement learning, particularly for approximating Q-values in **Q-learning** using neural networks.

Let’s break it down:

* * *

### **1\. Background: Q-Learning**

Q-Learning is a model-free reinforcement learning algorithm that seeks to learn the **Q-function**, which estimates the expected future rewards of taking an action  $a$  in a given state  $s$  and following the optimal policy thereafter:

$$
Q(s, a) = \text{expected return from } s \text{ taking } a
$$

The update rule is:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

* * *

### **2\. Problem: Q-Table Doesn’t Scale**

In large or continuous state spaces (like images or high-dimensional sensors), it's impractical to use a table to store Q-values for every state-action pair.

* * *

### **3\. Solution: Deep Q-Network (DQN)**

Instead of a table, a **neural network**  $Q(s, a; \theta)$  is used to approximate Q-values. This is called a **Deep Q-Network (DQN)**.

In **Naive DQN**, this is done in the simplest possible way, usually **without the stabilizing tricks** used in more advanced versions.

* * *

### **4\. Naive DQN Structure**

*   Input: Current state  $s$ 
    
*   Output: Q-values for all possible actions
    
*   Loss function: Based on the temporal difference (TD) error:
    

$$
L(\theta) = \left( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2
$$

* * *

### **5\. Limitations of Naive DQN**

Naive DQN can learn, but it suffers from **instability and divergence** due to:

*   **Correlated updates** (samples from consecutive time steps are similar)
    
*   **Moving targets** (the Q-network is both selecting and evaluating the action)
    
*   **Overestimation bias** (always taking max Q-values)
    

* * *

### **6\. Improvements Over Naive DQN**

The full DQN introduced by DeepMind in 2015 includes several improvements:

*   **Experience Replay**: Store past transitions in a buffer and sample randomly to break correlations.
    
*   **Target Network**: Maintain a separate network  $Q_{\text{target}}$  for computing the target Q-value, which is updated less frequently.
    
*   **Clipping rewards or normalizing inputs**
    

* * *

### **Summary**

| Aspect | Naive DQN |
| --- | --- |
| Q-function | Approximated with a neural network |
| Stability tricks | None (no replay, no target network) |
| Uses | Good for small problems; educational |
| Problems | Unstable learning, diverging Q-values |

* * *

Would you like a diagram or pseudocode to make this even clearer?

## Prompt:
HW4-1\_Naive DQN.ipynb

File

Explain this program

## Response:
Analyzed

```python
import nbformat

# Load the uploaded Jupyter Notebook
notebook_path = "/mnt/data/HW4-1_Naive DQN.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Extract all code cells
code_cells = [cell['source'] for cell in notebook_content['cells'] if cell['cell_type'] == 'code']
code_cells[:5]  # Show the first few cells for review
```

Result

undefined

Here's a breakdown of the **first part** of your uploaded notebook titled **"HW4-1\_Naive DQN.ipynb"**:

* * *

### 🔹 **Code Cell 1: Data Visualization**

This cell is not related to DQN. It’s a basic example of using matplotlib and numpy:

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(100)  # Generate 100 random numbers
mean_value = np.mean(data)   # Compute mean

# Plot the data
plt.scatter(range(len(data)), data, alpha=0.5)
plt.axhline(mean_value, color='red', linestyle='--', label='Mean')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('Scatter Plot with Mean Value')
plt.legend()
plt.show()
```

🟡 **Purpose**: Probably just a demo of plotting. Not part of the DQN logic.

* * *

### 🔹 **Code Cell 2: Environment Setup**

```python
from Gridworld import Gridworld
game = Gridworld(size=4, mode='static')
```

🟢 **Explanation**:

*   Loads a **custom Gridworld environment**, likely a small grid used for testing reinforcement learning agents.
    
*   `size=4`: 4x4 grid.
    
*   `mode='static'`: Static configuration (probably fixed reward/goal layout).
    

* * *

### 🔹 **Code Cell 3 & 4: Manual Game Interaction**

```python
game.display()            # Shows the current grid state
game.makeMove('d')        # Moves the agent (likely 'd' = right)
```

🟢 **Purpose**:

*   Allows you to visually test the environment manually before applying the DQN.
    

* * *

Would you like me to continue explaining the rest of the cells (especially the parts implementing the Naive DQN)?

## Prompt:
Continue

## Response:
Analyzed

```python
# Show the next batch of code cells for explanation
code_cells[5:10]
```

Result

undefined

Here's the continuation of the explanation for your **Naive DQN implementation**:

* * *

### 🔹 **Code Cells 5–8: More Environment Exploration**

```python
game.display()
game.reward()
game.board.render_np()
game.board.render_np().shape
```

🟢 **Purpose**:

*   Explore how the environment looks and what kind of state representation it provides.
    
*   `render_np()` likely returns a **flat NumPy array** used as input to the neural network.
    

* * *

### 🔹 **Code Cell 9: Model Definition and Initialization**

```python
import numpy as np
import torch
from Gridworld import Gridworld
from IPython.display import clear_output
import random
from matplotlib import pylab as plt

# Neural Network architecture
L1 = 64    # Input layer (flattened state)
L2 = 150   # First hidden layer
L3 = 100   # Second hidden layer
L4 = 4     # Output layer (Q-values for 4 actions)

model = torch.nn.Sequential(
    torch.nn.Linear(L1, L2),
    torch.nn.ReLU(),
    torch.nn.Linear(L2, L3),
    torch.nn.ReLU(),
    torch.nn.Linear(L3, L4)
)

# Loss and optimizer
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# RL hyperparameters
gamma = 0.9      # Discount factor
epsilon = 1.0    # Initial exploration rate
```

🔍 **Explanation**:

*   A simple **feedforward neural network** is used to approximate the Q-values.
    
*   The input size is 64, suggesting the environment provides an 8×8 state flattened to 64 features.
    
*   The output is 4 values – presumably for the 4 actions: up, down, left, right.
    
*   This is a classic **Naive DQN** setup: one model, no target network, no experience replay.
    

* * *

Would you like me to continue with the training loop and how actions are selected and updated? That will show how this model actually learns.

## Prompt:
Continue

## Response:
Analyzed

```python
# Show the next batch of code cells to continue explanation
code_cells[10:15]
```

Result

undefined

Here's the key part of your Naive DQN implementation — the **training loop** — along with an explanation of how your agent learns:

* * *

### 🔹 **Code Cell 10: Action Mapping**

```python
action_set = {
    0: 'u',  # up
    1: 'd',  # down
    2: 'l',  # left
    3: 'r'   # right
}
```

🔹 **Maps action indices (0–3)** to their corresponding movement directions in the Gridworld.

* * *

### 🔹 **Code Cell 11: Training Loop**

This is the core logic of **Naive DQN training**:

#### 🧠 **Overview**:

*   Trains for 1000 episodes (epochs)
    
*   Uses an ε-greedy strategy (ε decays over time)
    
*   No experience replay or target network
    

#### 🧩 **Key Elements**:

```python
epochs = 1000
losses = []

for i in range(epochs):
    game = Gridworld(size=4, mode='static')
    state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    state1 = torch.from_numpy(state_).float()
    status = 1
    while(status == 1):
        qval = model(state1)
        qval_ = qval.data.numpy()

        # Epsilon-greedy action selection
        if (random.random() < epsilon): 
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)

        action = action_set[action_]
        game.makeMove(action)

        # Get next state
        state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()

        # Estimate future rewards
        with torch.no_grad():
            newQ = model(state2)
        maxQ = torch.max(newQ)

        # Compute target Q-value
        Y = reward + (gamma * maxQ) if reward == -1 else reward
        Y = torch.Tensor([Y]).detach()

        # Current Q-value
        X = qval.squeeze()[action_]

        # Loss and backprop
        loss = loss_fn(X, Y)
        if i % 100 == 0:
            print(i, loss.item())
            clear_output(wait=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state1 = state2
        if abs(reward) == 10:
            status = 0

    losses.append(loss.item())

    if epsilon > 0.1:
        epsilon -= (1/epochs)  # decay epsilon

# Plot training loss
plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
```

* * *

### ✅ **What's Happening?**

| Part | Purpose |
| --- | --- |
| `qval = model(state1)` | Predict Q-values for current state |
| `action_ = argmax(qval)` | Choose best action (or random) |
| `Y = reward + γ * maxQ` | Compute the TD target |
| `loss = MSE(predicted, target)` | Train to reduce difference between predicted and target Q |
| `epsilon decay` | Decrease exploration over time |

* * *

### 🔹 **Code Cells 12–14**

These are unrelated to DQN; they test basic PyTorch gradient mechanics using a simple linear model. Likely used for learning or checking how gradients work.

* * *

Would you like me to summarize the entire Naive DQN process in plain steps or help modify this to include improvements like Experience Replay or a Target Network?

## Prompt:
Summarize the entire Naive DQN process.

## Response:
Here’s a clear **summary of the entire Naive DQN process** from your notebook:

* * *

🧠 Naive DQN: Summary
---------------------

### **1\. Environment Setup**

*   You use a custom **Gridworld environment** (`size=4`, `mode='static'`) to simulate a simple grid-based game.
    
*   The state is represented as a **flattened 64-dimensional vector** (4x4 grid with 4 channels).
    

* * *

### **2\. Neural Network Architecture**

*   A simple **feedforward neural network** approximates the Q-function  $Q(s, a)$ :
    
    *   Input: 64 features (flattened state)
        
    *   Hidden Layers: 150 and 100 neurons with ReLU
        
    *   Output: 4 values (Q-values for 4 actions: up, down, left, right)
        

* * *

### **3\. Training Loop**

For each of 1000 episodes:

#### **a. Initialization**

*   Reset the game.
    
*   Add noise to the state to improve generalization.
    
*   Set exploration rate `ε` high (starts at 1.0).
    

#### **b. Action Selection (ε-greedy)**

*   With probability ε: pick a random action (exploration)
    
*   Otherwise: pick the action with the **highest predicted Q-value** (exploitation)
    

#### **c. State Transition**

*   Execute the action in the environment.
    
*   Observe the new state and the reward.
    
*   Check if the episode is done.
    

#### **d. Q-Value Update (TD Learning)**

*   Predict Q-values for the new state.
    
*   Compute the **target Q-value**:
    
    $$
    Y = \text{reward} + \gamma \max_{a'} Q(s', a')
    $$
    
    unless it's a terminal state, in which case  $Y = \text{reward}$ .
    
*   Compute the loss (MSE) between predicted Q and target Q.
    
*   Backpropagate and update weights with **Adam optimizer**.
    

#### **e. Epsilon Decay**

*   Gradually decrease ε to reduce exploration over time (down to 0.1).
    

* * *

### **4\. Plot Training Loss**

*   The loss per episode is recorded and plotted to visualize learning progress.
    

* * *

⚠️ Limitations of Naive DQN
---------------------------

*   **No Experience Replay**: Each step is learned immediately, which leads to correlated updates.
    
*   **No Target Network**: The same model is used to select and evaluate actions, which can cause instability.
    
*   **Basic ε-decay**: Simple linear schedule, no adaptive or smart decay.
    

* * *

✅ Educational Value
-------------------

Despite its limitations, Naive DQN captures the **core idea of using deep learning for reinforcement learning** and is great for:

*   Learning the RL pipeline
    
*   Testing environments
    
*   Extending into full DQN with stabilizing tricks
    

* * *

Would you like a visual diagram of the DQN architecture or help upgrading this to a more robust version (with replay buffer, target network, etc.)?

## Prompt:
Basic DQN implementation for an easy environment

## Response:
Here’s a **basic DQN implementation** in PyTorch for an easy environment: `CartPole-v1` from OpenAI Gym. This includes **experience replay** and a **target network**, making it more stable than Naive DQN but still easy to follow.

* * *

🧪 Requirements
---------------

```bash
pip install gym torch numpy matplotlib
```

* * *

🧠 Basic DQN Implementation for CartPole
----------------------------------------

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Neural Network for Q-value approximation
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# Experience Replay Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, transition):
        self.buffer.append(transition)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# Initialize environment, model, etc.
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)
epsilon = EPSILON_START

rewards_all = []

# Training loop
for episode in range(EPISODES):
    state = env.reset()[0]
    total_reward = 0

    done = False
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones).unsqueeze(1)

            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + GAMMA * max_next_q_values * (~dones)

            loss = nn.MSELoss()(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update target network
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    rewards_all.append(total_reward)

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# Plot rewards
plt.plot(rewards_all)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN on CartPole')
plt.show()
```

* * *

### ✅ Features Included

*   **Neural network** for Q-value approximation
    
*   **Experience replay**
    
*   **Target network**
    
*   **Epsilon-greedy exploration**
    

* * *

Would you like a simplified or modified version for another custom environment like Gridworld?

## Prompt:
Experience Replay Buffer

## Response:
### 🧠 **Experience Replay Buffer** – Explained Simply

**Experience Replay** is a technique used in Deep Q-Learning to stabilize training and improve sample efficiency.

* * *

### 🔁 **Why Use It?**

In naive DQN:

*   You train on consecutive samples, which are **highly correlated**.
    
*   This makes learning unstable and inefficient.
    

With **experience replay**:

*   You **store past experiences** (state, action, reward, next\_state, done) in a buffer.
    
*   Then you **sample randomly** from this buffer to train the model.
    

This helps:

*   Break correlations between sequential samples.
    
*   Reuse experiences multiple times.
    
*   Reduce variance and improve convergence.
    

* * *

### 🧱 **Structure of the Buffer**

A basic experience replay buffer is just a **fixed-size queue (deque)** where each element is a transition:

```python
(state, action, reward, next_state, done)
```

* * *

### ✅ **Minimal Implementation in Python**

```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # Automatically discards old experiences

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
```

* * *

### 🧪 **How to Use It in DQN**

```python
# Initialize
buffer = ReplayBuffer(capacity=10000)

# Store a transition
buffer.push((state, action, reward, next_state, done))

# Sample a batch for training
if len(buffer) >= batch_size:
    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
```

* * *

Would you like an illustration of how this buffer interacts with training steps or how to combine it with prioritized replay?

## Prompt:
Explain buffer interacts with training steps and combine it with prioritized replay.

## Response:
Great — let’s go step by step:

* * *

🔁 **1\. How the Replay Buffer Interacts with Training Steps**
--------------------------------------------------------------

### 🔹 Every Training Step Includes:

#### 1\. **Collecting Experience**

Each time the agent takes a step in the environment, you store a transition:

```python
transition = (state, action, reward, next_state, done)
buffer.push(transition)
```

#### 2\. **Sampling a Batch**

Once the buffer has enough data, you sample a **mini-batch** for training:

```python
batch = buffer.sample(batch_size)
```

#### 3\. **Training on Sampled Batch**

*   Extract tensors (states, actions, etc.)
    
*   Predict Q-values and compute target Q-values
    
*   Calculate loss (usually MSE)
    
*   Backpropagate and update model weights
    

* * *

### 📦 **Diagram of the Flow**

```
 Environment Step
      ↓
( state, action, reward, next_state, done )
      ↓
   PUSH into Replay Buffer
      ↓
 SAMPLE Mini-Batch from Buffer
      ↓
 Train Model on Random Batch
```

This breaks temporal correlation and improves stability in training.

* * *

⭐️ **2\. Prioritized Experience Replay (PER)**
----------------------------------------------

### 🧠 **Motivation**

Not all experiences are equally useful. Prioritized Replay gives more attention to **“surprising”** or **“important”** transitions (e.g., with higher TD error).

* * *

### 🔹 Core Ideas of PER:

*   **Priority Score**: Each transition gets a priority (typically based on TD-error).
    
*   **Sampling Probability**: Transitions are sampled with probability proportional to their priority.
    
*   **Bias Correction**: Use **importance-sampling weights** to correct the bias this introduces.
    

* * *

### ⚙️ **Simple PER Workflow**

```python
1. Calculate TD Error = |Q_pred - Q_target|
2. Set priority = TD error + small constant (e.g., 1e-5)
3. Sample batch using weighted probability based on priorities
4. Scale the loss using importance-sampling weights (to reduce bias)
```

* * *

### 🛠️ **Simplified PER Buffer (Conceptual Code)**

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha

    def push(self, transition, td_error):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            # Replace oldest
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(transition)
            self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize

        return samples, indices, torch.FloatTensor(weights)

    def update_priority(self, indices, new_td_errors):
        for idx, td in zip(indices, new_td_errors):
            self.priorities[idx] = (abs(td) + 1e-5) ** self.alpha
```

* * *

### 🧪 Training Integration with PER:

1.  After computing the TD error for each sampled transition:
    
    *   Use it to compute priority.
        
    *   Update the buffer with new priorities.
        
2.  When computing loss:
    
    *   Multiply it by the importance-sampling weights.
        

* * *

### ✅ Benefits of PER:

*   Speeds up learning by focusing on impactful experiences
    
*   Makes better use of limited memory
    

* * *

Would you like a full runnable PyTorch DQN example using Prioritized Experience Replay?

## Prompt:
Give me a short report.

## Response:
Sure! Here's a **short report** summarizing the key concepts and methods related to **Experience Replay** and **Prioritized Experience Replay (PER)** in Deep Q-Learning.

* * *

📘 Short Report: Experience Replay and Prioritized Experience Replay in DQN
---------------------------------------------------------------------------

### 🎯 Objective

To stabilize and improve the learning process in Deep Q-Networks (DQN) by reusing past experiences in a more efficient and targeted way.

* * *

### 🔁 Experience Replay

**Experience Replay** stores past agent interactions in a memory buffer:

*   Each interaction is a tuple:
    
    $$
    (state, action, reward, next\_state, done)
    $$
    
*   During training, random mini-batches are sampled from this buffer.
    

#### ✅ Benefits:

*   Breaks temporal correlations between consecutive states.
    
*   Improves data efficiency by reusing old experiences.
    
*   Stabilizes learning by reducing variance in updates.
    

#### 🔧 Implementation (Basic):

*   A `deque` stores transitions.
    
*   When enough samples are collected, random batches are used to train the Q-network.
    

* * *

### ⭐️ Prioritized Experience Replay (PER)

**PER** improves upon uniform sampling by prioritizing experiences with **higher learning potential**.

#### 🧠 Key Idea:

*   Prioritize transitions with large **TD errors** (temporal difference between predicted and target Q-values).
    
*   These samples are likely to yield greater updates to the Q-network.
    

#### ⚙️ Mechanism:

*   Assign each transition a priority score:
    
    $$
    p_i = (|\text{TD-error}_i| + \varepsilon)^\alpha
    $$
    
*   Sample based on this score (not uniformly).
    
*   Use **importance-sampling weights** to correct for bias.
    

#### ✅ Benefits:

*   Focuses learning on the most informative experiences.
    
*   Accelerates convergence and improves final performance.
    

* * *

### 📌 Summary of Differences

| Feature | Experience Replay | Prioritized Replay |
| --- | --- | --- |
| Sampling | Uniform | Based on TD-error priority |
| Data reuse | Random | Targeted (important samples) |
| Stability | Moderate | Higher |
| Implementation complexity | Simple | More involved |

* * *

### 🧩 Use Case in DQN Training

1.  **Interact with the environment** and collect transitions.
    
2.  **Store transitions** in the buffer (with priority in PER).
    
3.  **Sample a mini-batch** (uniformly or by priority).
    
4.  **Train the model** using predicted vs. target Q-values.
    
5.  **Update priorities** in PER after computing TD-errors.
    

* * *

### 🧪 Conclusion

Experience Replay is crucial for effective deep reinforcement learning. While standard replay buffers offer stability, Prioritized Experience Replay enhances learning by focusing on high-impact transitions. Both techniques are widely used in modern RL algorithms.
