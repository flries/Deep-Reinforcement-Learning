## **Double DQN**

Double DQN (Double Deep Q-Network) improves the stability and accuracy of Q-learning by **reducing the overestimation of Q-values** that standard DQN suffers from.

* * *

### 🧠 Key Components

1.  **Two Networks**:
    
    *   **Online (Policy) Network** `Q(s, a; θ)`: used to select actions.
        
    *   **Target Network** `Q(s, a; θ⁻)`: used to evaluate actions.
        
    *   Both networks share the same architecture but maintain separate weights.
        
2.  **Target Calculation (Core Difference)**:
    
    *   **DQN**:  
         $Q_{\text{target}} = r + \gamma \cdot \max_a Q(s', a; \theta^-)$ 
        
    *   **Double DQN**:  
         $Q_{\text{target}} = r + \gamma \cdot Q(s', \arg\max_a Q(s', a; \theta); \theta^-)$   
        → Use **online net** to choose the best action, but **target net** to evaluate it.
        
3.  **Target Network Update**:
    
    *   Periodically copy weights from the online network:
        
        ```python
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        ```
        
4.  **Training Loop**:
    
    *   Initialize environment and state.
        
    *   Choose actions with ε-greedy policy.
        
    *   Store transition: (state, action, reward, next\_state).
        
    *   Sample minibatch from replay memory.
        
    *   Compute Q-values and targets using Double DQN logic.
        
    *   Minimize loss between predicted and target Q-values.
        

* * *

### 🔄 Summary of Steps

1.  Initialize `policy_net` and `target_net` with the same architecture and weights.
    
2.  For each episode:
    
    *   Select action via ε-greedy from `policy_net`.
        
    *   Execute action in environment.
        
    *   Observe reward and next state.
        
    *   Compute target using:
        
        ```python
        next_action = policy_net(next_state).argmax(1)
        target_q = target_net(next_state).gather(1, next_action.unsqueeze(1))
        ```
        
    *   Compute loss and update `policy_net`.
        
3.  Periodically update `target_net` to follow `policy_net`.
    

* * *

### ✅ Benefits

*   Reduces overestimation of Q-values.
    
*   Improves training stability.
    
*   Achieves better performance in many environments compared to vanilla DQN.

### **Dueling DQN**

**Dueling DQN** enhances the learning efficiency of Deep Q-Networks by **separating the estimation of state value and action advantage**—especially beneficial when it's not necessary to evaluate all actions.

* * *

### 🧠 Key Architecture Innovation

Instead of outputting Q-values directly, the network splits into two separate streams after the shared feature layers:

1.  **Value stream**: Estimates how good it is to be in a state  $V(s)$ .
    
2.  **Advantage stream**: Estimates the relative benefit of each action  $A(s, a)$ .
    

They are combined to produce Q-values:

$$
Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)
$$

This structure helps the model **better evaluate states**, even when the advantage of specific actions is small or unclear.


* * *

### 🔄 Training Process

1.  **Initialize** the dueling architecture model and target model with same parameters.
    
2.  **Action selection**: Use ε-greedy based on Q-values.
    
3.  **Step through the environment**: Collect (state, action, reward, next\_state).
    
4.  **Target computation** (can be combined with Double DQN):
    
    ```python
    next_action = policy_net(next_state).argmax(1)
    target_q = target_net(next_state).gather(1, next_action.unsqueeze(1))
    ```
    
5.  **Loss function**: MSE between predicted Q and target Q.
    
6.  **Backpropagate and optimize**.
    
7.  **Update target network** periodically.
    

* * *

### ✅ Benefits

*   **More accurate value estimation**: Focuses learning on the quality of states.
    
*   **Improved performance in large action spaces**.
    
*   **Works well with Double DQN** for additional stability.
    

* * *

Would you like a combined explanation for **Dueling + Double DQN**, which is commonly used in practice?

* * *

🔰 Baseline: Basic DQN
----------------------

**Key idea**: Approximates the Q-function  $Q(s, a)$  using a neural network.

### 🔻 Limitations:

*   **Overestimates Q-values** due to max operator in target calculation.
    
*   **Learns slowly** in states where actions don't affect the outcome (inefficient state evaluation).
    

* * *

✅ Double DQN: Reducing Overestimation
-------------------------------------

### 🔍 Problem It Solves:

*   Basic DQN uses:
    
    $$
    Q_{\text{target}} = r + \gamma \cdot \max_a Q(s', a; \theta^-)
    $$
    
    → The `max` selects the highest Q-value even if it's overestimated.
    

### 🧠 How It Works:

*   **Decouples action selection and evaluation**:
    
    $$
    Q_{\text{target}} = r + \gamma \cdot Q(s', \arg\max_a Q(s', a; \theta); \theta^-)
    $$
    *   `policy_net` chooses the action (less prone to noise).
        
    *   `target_net` evaluates that action.
        

### 📈 Improvement Over Basic DQN:

*   More **accurate Q-value targets**.
    
*   Reduces bias → **better convergence** and **stability**.
    

* * *

✅ Dueling DQN: Better State Value Estimation
--------------------------------------------

### 🔍 Problem It Solves:

*   Basic DQN learns Q-values for all actions even when **actions don't affect outcomes**.
    
*   Wastes capacity learning irrelevant action differences.
    

### 🧠 How It Works:

*   Splits the network:
    
    *   One stream estimates **state value**  $V(s)$ 
        
    *   Another estimates **advantage**  $A(s, a)$ 
        
    *   Combines them:
        
        $$
        Q(s, a) = V(s) + (A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'))
        $$
        

### 📈 Improvement Over Basic DQN:

*   Learns **which states are valuable**, independent of specific actions.
    
*   Improves **data efficiency**, especially in large or sparse action spaces.
    

* * *

🔁 Summary Table
----------------

| Feature | Basic DQN | Double DQN | Dueling DQN |
| --- | --- | --- | --- |
| Q-Value Estimation | Direct max | Decoupled selection/evaluation | Split into Value + Advantage |
| Overestimation Bias | High | Low | Moderate |
| State Value Awareness | No | No | Yes |
| Architectural Change | None | No (uses 2 networks) | Yes (2-stream architecture) |
| Best Use Case | Simple tasks | Tasks with noisy Q-values | Tasks where state value is key |