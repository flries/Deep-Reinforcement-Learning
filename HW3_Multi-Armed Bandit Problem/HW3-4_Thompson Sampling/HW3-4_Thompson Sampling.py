# Thompson Sampling Algorithm for Multi-Armed Bandit Problem

import numpy as np
import matplotlib.pyplot as plt

# 1. Set up the problem
np.random.seed(42)

num_arms = 5
true_means = np.random.normal(0, 1, num_arms)  # True mean rewards
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
plt.xlabel('Iteration')
plt.ylabel('Selected Arm')
plt.grid(True)
plt.show()
