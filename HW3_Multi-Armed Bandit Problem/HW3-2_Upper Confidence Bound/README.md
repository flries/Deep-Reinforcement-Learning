1\. LaTeX Formula for UCB
------------------------------------

The UCB formula for selecting arm  $a$  at time  $t$  is:

$$
a_t = \arg\max_{a} \left( \hat{\mu}_a + \sqrt{ \frac{2 \ln t}{N_a} } \right)
$$

Where:

*    $\hat{\mu}_a$  is the estimated average reward for arm  $a$ .
*    $N_a$  is the number of times arm  $a$  has been selected.
*    $t$  is the current time step (iteration).
    

* * *

2\. Key Logic / Analysis
------------------------

*   **Optimism in the Face of Uncertainty**:  
    UCB favors arms that either have high estimated rewards or have been tried less (i.e., high uncertainty).
*   **Exploration naturally decays**:  
    As arms are played more often, the uncertainty term shrinks, making the algorithm exploit the best-known arms over time.
*   **Balance**:  
    Early on, UCB explores all arms to gain confidence, but over time it leans toward the arms with the highest rewards.
*   **Key idea**:  
    Prefer arms with a combination of high reward and high uncertainty.

* * *

3\. Code + Charts
-----------------
```python
# Upper Confidence Bound (UCB1) Algorithm for Multi-Armed Bandit Problem
# Initially, select each arm once
arm = t - 1
else:
ucb_values = estimated_means + np.sqrt(2 * np.log(t) / (arm_counts + 1e-5))
arm = np.argmax(ucb_values)

reward = np.random.normal(true_means[arm], 1)

arm_counts[arm] += 1
# Update the mean estimate incrementally
estimated_means[arm] += (reward - estimated_means[arm]) / arm_counts[arm]

rewards.append(reward)
selected_arms.append(arm)

# 3. Plot the accumulated rewards and convergence
cumulative_rewards = np.cumsum(rewards)
average_rewards = cumulative_rewards / (np.arange(num_iterations) + 1)

plt.figure(figsize=(12, 6))
plt.plot(average_rewards, label='Average Reward')
plt.axhline(np.max(true_means), color='r', linestyle='--', label='Optimal Reward')
plt.title('UCB: Convergence Over Time')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.show()

# 4. Spatial and Temporal Analysis

# Spatial: Number of times each arm was selected
plt.figure(figsize=(10, 5))
plt.bar(range(num_arms), arm_counts)
plt.title('Spatial Analysis: Number of Selections per Arm (UCB)')
plt.xlabel('Arm Index')
plt.ylabel('Number of Times Selected')
plt.grid(True)
plt.show()

# Temporal: Choices over time
plt.figure(figsize=(12, 4))
plt.scatter(range(num_iterations), selected_arms, s=10)
plt.title('Temporal Analysis: Arm Selection Over Time (UCB)')
plt.xlabel('Iteration')
plt.ylabel('Selected Arm')
plt.grid(True)
plt.show()

```
*   **Average cumulative rewards** over time (convergence).
![HW3-2_Upper Confidence Bound_Result](https://github.com/user-attachments/assets/f890e444-60d3-4d7a-aa54-d3ea4fdbfc30)
* * *

4\. Spatial and Temporal Analysis
---------------------------------
![HW3-2_Upper Confidence Bound_Spatial](https://github.com/user-attachments/assets/3c72bcb2-5135-48b6-8347-788686256431)
**Spatial Analysis:**

*   Shows how often each arm was chosen.
*   Ideally, the best arm (highest true mean) gets selected the most after sufficient trials.

![HW3-2_Upper Confidence Bound_Temporal](https://github.com/user-attachments/assets/e1b1e537-4bf0-4caf-996d-326509d570a7)
**Temporal Analysis:**

*   Early iterations: More variation and exploration.
*   Later iterations: Clear dominance of the best arm, fewer explorations.
