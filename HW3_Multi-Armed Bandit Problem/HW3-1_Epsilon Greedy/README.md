1\. LaTeX Formula for Epsilon-Greedy
------------------------------------

The decision rule at each step  $t$  is:

$$
a_t = \begin{cases} \text{random arm}, & \text{with probability } \varepsilon \quad (\text{exploration}) \\\\ \arg\max_{a} \hat{\mu}_a, & \text{with probability } 1 - \varepsilon \quad (\text{exploitation}) \end{cases}
$$

Where:

*    $a_t$  is the selected arm at time  $t$ .
*    $\hat{\mu}_a$  is the estimated mean reward for arm  $a$ .
*    $\varepsilon$  is a small constant (like 0.1).
    

* * *

2\. Key Logic / Analysis
------------------------

*   **Exploration**: Randomly try arms to gather more information and avoid premature convergence on suboptimal arms.
*   **Exploitation**: Choose the best-known arm to maximize rewards based on current estimates.
*   **Balance**:  $\varepsilon$  controls how much you explore; smaller  $\varepsilon$  → more exploitation.
*   **Learning**: Update the estimated rewards incrementally after each pull.
    

Thus, epsilon-greedy **gradually learns** the best arm but ensures **occasional exploration** to avoid getting stuck with suboptimal choices.

* * *

3\. Code + Charts
-----------------
```python
# Epsilon-Greedy Algorithm for Multi-Armed Bandit Problem

import numpy as np
import matplotlib.pyplot as plt

# 1. Set up the problem
np.random.seed(42)

num_arms = 5
true_means = np.random.normal(0, 1, num_arms) # True mean rewards
num_iterations = 1000

epsilon = 0.1 # Exploration probability

estimated_means = np.zeros(num_arms) # Initial estimated rewards
arm_counts = np.zeros(num_arms) # Times each arm has been selected

rewards = []
selected_arms = []

# 2. Run Epsilon-Greedy Algorithm
for t in range(1, num_iterations + 1):
if np.random.rand() < epsilon:
arm = np.random.randint(num_arms) # Explore
else:
arm = np.argmax(estimated_means) # Exploit

reward = np.random.normal(true_means[arm], 1) # Simulated reward

arm_counts[arm] += 1
# Update estimated mean using incremental formula
estimated_means[arm] += (reward - estimated_means[arm]) / arm_counts[arm]

rewards.append(reward)
selected_arms.append(arm)

```
*   **Average cumulative rewards** over time (convergence).
![HW3-1_Epsilon Greedy_Result](https://github.com/user-attachments/assets/c4cfb449-1096-4214-9c18-83a1b8eea2c6)
* * *

4\. Spatial and Temporal Analysis
---------------------------------
![HW3-1_Epsilon Greedy_Spatial](https://github.com/user-attachments/assets/fd3dda17-0494-47c8-bde3-70fe53d385b4)
**Spatial Analysis:**

*   Shows which arms were chosen the most.
*   Ideally, the best arm should dominate after enough exploration.
  
![HW3-1_Epsilon Greedy_Temporal](https://github.com/user-attachments/assets/fe82d075-f2c9-435a-acc3-800f5282cddd)
**Temporal Analysis:**

*   Early stages: More random, many arms explored.
*   Later stages: Choices concentrate on the best arm, but occasional random picks still appear due to  $\varepsilon$ .
