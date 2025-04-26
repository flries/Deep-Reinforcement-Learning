# Softmax Action Selection Algorithm for Multi-Armed Bandit Problem

import numpy as np
import matplotlib.pyplot as plt

# 1. Set up the problem
np.random.seed(42)

num_arms = 5
true_means = np.random.normal(0, 1, num_arms)  # True mean rewards
num_iterations = 1000

temperature = 0.5  # Temperature parameter for softmax

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
    selected_arms.append(arm)

# 3. Plot the accumulated rewards and convergence
cumulative_rewards = np.cumsum(rewards)
average_rewards = cumulative_rewards / (np.arange(num_iterations) + 1)

plt.figure(figsize=(12, 6))
plt.plot(average_rewards, label='Average Reward')
plt.axhline(np.max(true_means), color='r', linestyle='--', label='Optimal Reward')
plt.title('Softmax: Convergence Over Time')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.show()

# 4. Spatial and Temporal Analysis

# Spatial: Number of times each arm was selected
plt.figure(figsize=(10, 5))
plt.bar(range(num_arms), arm_counts)
plt.title('Spatial Analysis: Number of Selections per Arm (Softmax)')
plt.xlabel('Arm Index')
plt.ylabel('Number of Times Selected')
plt.grid(True)
plt.show()

# Temporal: Choices over time
plt.figure(figsize=(12, 4))
plt.scatter(range(num_iterations), selected_arms, s=10)
plt.title('Temporal Analysis: Arm Selection Over Time (Softmax)')
plt.xlabel('Iteration')
plt.ylabel('Selected Arm')
plt.grid(True)
plt.show()
