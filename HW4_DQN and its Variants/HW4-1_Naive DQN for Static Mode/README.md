### **Naive DQN Structure**

*   Input: Current state  $s$ 
    
*   Output: Q-values for all possible actions
    
*   Loss function: Based on the temporal difference (TD) error:
    

$$
L(\theta) = \left( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2
$$

* * *

### **Limitations of Naive DQN**

Naive DQN can learn, but it suffers from **instability and divergence** due to:

*   **Correlated updates** (samples from consecutive time steps are similar)
    
*   **Moving targets** (the Q-network is both selecting and evaluating the action)
    
*   **Overestimation bias** (always taking max Q-values)
    

* * *

### **Improvements Over Naive DQN**

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
