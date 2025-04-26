1\. LaTeX Formula for Thompson Sampling
------------------------------------

At each time  $t$ , for each arm  $a$ :
$$
\theta_a \sim \text{Posterior}(a)
$$

Then select:
$$
a_t = \arg\max_a \theta_a
$$

Where:

*    $\theta_a$  is a sample drawn from the posterior distribution of arm  $a$ 's reward.
*    $a_t$  is the arm selected at time  $t$ .
    

* * *

2\. Key Logic / Analysis
------------------------

*   **Bayesian approach**:  
    Instead of just using point estimates, Thompson Sampling maintains a probability distribution over the expected reward of each arm.
*   **Sampling for decision-making**:  
    At each step, sample a possible reward from each arm's distribution and select the arm with the highest sampled reward.
*   **Natural exploration-exploitation balance**:  
    Arms with greater uncertainty are more likely to be sampled, encouraging exploration.
*   **Self-tuning**:  
    As more data is collected, the posterior distribution becomes sharper, leading to more confident exploitation.

* * *

3\. Code + Charts
-----------------
```python
# Thompson Sampling Algorithm for Multi-Armed Bandit Problem

import numpy as np
import matplotlib.pyplot as plt

# 1. Set up the problem
np.random.seed(42)

num_arms = 5
true_means = np.random.normal(0, 1, num_arms) # True mean rewards
num_iterations = 1000

# Initialize parameters for Thompson Sampling (Gaussian assumption)
estimated_means = np.zeros(num_arms)
arm_counts = np.zeros(num_arms)

rewards = []
selected_arms = []

# 2. Run Thompson Sampling Algorithm
for t in range(1, num_iterations + 1):
sampled_means = np.random.normal(estimated_means, 1 / (np.sqrt(arm_counts + 1e-5)))
arm = np.argmax(sampled_means)

reward = np.random.normal(true_means[arm], 1)

arm_counts[arm] += 1
# Update the estimated mean incrementally
estimated_means[arm] += (reward - estimated_means[arm]) / arm_counts[arm]

rewards.append(reward)
selected_arms.append(arm)

# 3. Plot the accumulated rewards and convergence
cumulative_rewards = np.cumsum(rewards)
average_rewards = cumulative_rewards / (np.arange(num_iterations) + 1)

plt.figure(figsize=(12, 6))
plt.plot(average_rewards, label='Average Reward')
plt.axhline(np.max(true_means), color='r', linestyle='--', label='Optimal Reward')
plt.title('Thompson Sampling: Convergence Over Time')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.show()

# 4. Spatial and Temporal Analysis

# Spatial: Number of times each arm was selected
plt.figure(figsize=(10, 5))
plt.bar(range(num_arms), arm_counts)
plt.title('Spatial Analysis: Number of Selections per Arm (Thompson Sampling)')
plt.xlabel('Arm Index')
plt.ylabel('Number of Times Selected')
plt.grid(True)
plt.show()

# Temporal: Choices over time
plt.figure(figsize=(12, 4))
plt.scatter(range(num_iterations), selected_arms, s=10)
plt.title('Temporal Analysis: Arm Selection Over Time (Thompson Sampling)')
```
*   **Average cumulative rewards** over time (convergence).
![HW3-4_Thompson Sampling_Result](https://github.com/user-attachments/assets/548d4ade-6d6c-41e8-a5ca-a30344cd5e19)

* * *

4\. Spatial and Temporal Analysis
---------------------------------
![HW3-4_Thompson Sampling_Spatial](https://github.com/user-attachments/assets/b6801a2a-30ab-4750-bead-baf1fd076722)
**Spatial Analysis:**

*   Displays how often each arm was selected.
*   The best arm should dominate after sufficient exploration.
  
![HW3-4_Thompson Sampling_Temporal](https://github.com/user-attachments/assets/2ee9e05f-9261-4720-9049-666f7243c044)
**Temporal Analysis**:

*   Early rounds show random exploration.
*   Over time, the selections should focus primarily on the arm with the best true reward.
