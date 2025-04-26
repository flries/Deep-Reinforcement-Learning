1\. LaTeX Formula for Softmax
------------------------------------

In Softmax action selection, the probability  $P(a)$  of selecting arm  $a$  is:

$$
P(a) = \frac{\exp\left(\frac{\hat{\mu}_a}{\tau}\right)}{\sum_{b=1}^K \exp\left(\frac{\hat{\mu}_b}{\tau}\right)}
$$

Where:

*   $\hat{\mu}_a$  is the estimated average reward of arm  $a$ .
*   $\tau$  (temperature) controls how random the selection is:
    *   High  $\tau$  → more uniform probabilities (more exploration).
    *   Low  $\tau$  → more greedy (exploitation).
    

* * *

2\. Key Logic / Analysis
------------------------

*   **Smooth exploration**:  
    Instead of hard choosing the best arm, Softmax assigns probabilities to each arm depending on their estimated rewards.
*   **Temperature controls randomness**:  
    A **high temperature** means more exploration; a **low temperature** means more focused exploitation.
*   **Advantages**:
    *   More balanced trade-off between exploration and exploitation.
    *   Avoids getting stuck on local optima compared to Epsilon-Greedy.
*   **Drawbacks**:
    *   Sensitive to the temperature parameter choice.

* * *

3\. Code + Charts
-----------------
```python
# Softmax Action Selection Algorithm for Multi-Armed Bandit Problem

import numpy as np
import matplotlib.pyplot as plt

# 1. Set up the problem
np.random.seed(42)

num_arms = 5
true_means = np.random.normal(0, 1, num_arms) # True mean rewards
num_iterations = 1000

temperature = 0.5 # Temperature parameter for softmax

estimated_means = np.zeros(num_arms)
arm_counts = np.zeros(num_arms)

rewards = []
selected_arms = []

# 2. Run Softmax Algorithm
for t in range(1, num_iterations + 1):
# Calculate softmax probabilities
exp_estimates = np.exp(estimated_means / temperature)
softmax_probs = exp_estimates / np.sum(exp_estimates)

# Select an arm according to softmax probabilities
arm = np.random.choice(np.arange(num_arms), p=softmax_probs)

reward = np.random.normal(true_means[arm], 1)

arm_counts[arm] += 1
# Update the estimated mean incrementally
estimated_means[arm] += (reward - estimated_means[arm]) / arm_counts[arm]

rewards.append(reward)
```
*   **Average cumulative rewards** over time (convergence).
* * *

4\. Spatial and Temporal Analysis
---------------------------------

**Spatial Analysis:**

*   Shows how often each arm was picked.
*   Arms with better estimated rewards are selected more frequently.

**Temporal Analysis:**

*   Early on: More diversity (high exploration if  $\tau$  is reasonable).
*   Later: Gradually stabilizes, focusing more on the best arms.