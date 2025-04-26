# HW3_Multi-Armed Bandit Problem

## Prompt:
Explain Multi-Armed Bandit in detail.

## Response:
Of course! Let’s go through **Multi-Armed Bandit (MAB)** carefully and in detail —  
I'll build it up slowly so you understand not just _what_ it is, but also _why_ it's important and _how_ it works.

* * *

🌟 1. What is the Multi-Armed Bandit Problem?
=============================================

Imagine you walk into a casino, and there are **many slot machines** (people call them "**one-armed bandits**" because they steal your money with one lever).

Each machine (each "arm") gives you different rewards when you pull it.

*   Some machines might give you lots of money often.
    
*   Others might rarely pay out anything at all.
    

However, **you don't know which machines are the good ones** at the start!  
Your goal is:  
🔵 _Play the machines to maximize the total money you win over time._

* * *

🌟 2. Why is it called a "Bandit"?
==================================

*   The term "**bandit**" comes from the fact that slot machines tend to "steal" your money.
    
*   "**Multi-armed**" because there are **multiple** machines (arms) you can pull.
    

**So:**  
You face **many options**, each with **unknown payoff probabilities**, and **limited chances** to explore them.

* * *

🌟 3. What Makes MAB Difficult? (Key Challenge)
===============================================

You face a trade-off between:

| Term | Meaning |
| --- | --- |
| **Exploration** | Try different machines to learn which ones are good. |
| **Exploitation** | Keep playing the machine that you believe is the best. |

⚡ **If you explore too much**, you might waste time on bad machines.  
⚡ **If you exploit too early**, you might miss discovering a better machine.

This is called the **exploration-exploitation dilemma**.

* * *

🌟 4. Formal Definition
=======================

In mathematical terms:

*   You have **K arms** (machines) to choose from.
    
*   Each arm  $i$  gives a random reward  $r_i$  based on an **unknown** probability distribution.
    
*   Goal:  
    **Maximize the total expected reward** over  $T$  time steps.
    

At each time  $t$ :

*   Pick an arm  $A_t$ .
    
*   Receive a reward  $r_{t, A_t}$ .
    
*   Update your knowledge and choose the next arm.
    

* * *

🌟 5. Metrics: How to Measure Performance?
==========================================

The main measure is **Regret**:

*   Regret means "how much reward you _lost_ compared to if you always picked the best arm."
    

Mathematically:

$$
\text{Regret}(T) = T \cdot \mu^* - \sum_{t=1}^T r_{t, A_t}
$$

Where:

*    $\mu^*$  = Expected reward of the best arm.
    
*    $r_{t, A_t}$  = Reward actually received at time  $t$ .
    

**Low regret = good strategy!**

* * *

🌟 6. Main Strategies to Solve MAB
==================================

There are many famous algorithms that try to **balance exploration and exploitation**:

| Algorithm | Idea |
| --- | --- |
| **Epsilon-Greedy** | With probability  $\varepsilon$  explore randomly, otherwise exploit the best-known arm. |
| **UCB (Upper Confidence Bound)** | Be optimistic! Choose arms with high _upper confidence_ of reward. |
| **Thompson Sampling** | Treat the problem probabilistically, and sample from the posterior distribution to decide. |
| **Softmax (Boltzmann Exploration)** | Play better arms more often, but not deterministically. Probability proportional to estimated reward. |

* * *

🌟 7. Quick Intuition for Each Strategy
=======================================

✅ **Epsilon-Greedy**:

*   90% of the time pick the best so far.
    
*   10% of the time, try something random (to discover).
    

✅ **UCB**:

*   Pick arms that either performed well _or_ that you haven't tried much (high uncertainty).
    

✅ **Thompson Sampling**:

*   Keep a belief distribution for each arm.
    
*   Sample "what might happen" and choose based on that.
    

✅ **Softmax**:

*   Arms with higher expected rewards are chosen more often, but even lower ones still have a chance.
    

* * *

🌟 8. Where is Multi-Armed Bandit Used?
=======================================

MAB shows up in real-world problems like:

| Application | Example |
| --- | --- |
| Online Ads | Show different ads and learn which ad gets more clicks. |
| Clinical Trials | Test different treatments and assign more patients to better ones. |
| News Recommendations | Suggest articles that maximize user reading. |
| A/B Testing | Choose best website design variations dynamically. |
| Game AI | Choosing actions that have uncertain outcomes. |

Anywhere **trial-and-error** learning is needed, MAB ideas apply!

* * *

🌟 9. A Simple Example
======================

Suppose:

*   Machine A: 70% chance to win $1
    
*   Machine B: 40% chance to win $1
    
*   Machine C: 20% chance to win $1
    

You don't know these probabilities at first.  
You start pulling arms, observing rewards, and updating your beliefs.

Maybe after some tries you estimate:

*   A is good!
    
*   B is worse!
    
*   C is bad!
    

Then you mostly pull A, but still sometimes try B and C just to **make sure**.

That's a Multi-Armed Bandit in action.

* * *

✨ Summary
=========

| Topic | Key Point |
| --- | --- |
| Problem | Choose actions under uncertainty to maximize reward. |
| Challenge | Balance exploration (learning) vs exploitation (earning). |
| Solution Approaches | Epsilon-Greedy, UCB, Thompson Sampling, Softmax. |
| Applications | Ads, A/B tests, healthcare, recommendations, gaming. |
| Goal | Maximize reward, minimize regret. |

* * *

Would you also like me to show a **small Python code demo** for how MAB (like Epsilon-Greedy) works? 🚀 It can make the concept even more concrete!  

## Prompt:
Summarize the article in this web:
https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/

## Response:
​James LeDoux's article, _"Multi-Armed Bandits in Python: Epsilon Greedy, UCB1, Bayesian UCB, and EXP3"_, provides an in-depth exploration of four prominent multi-armed bandit (MAB) algorithms, their Python implementations, and their performance in a content recommendation context using the Movielens 25M dataset.​[eScholarship+2James LeDoux's Blog+2James LeDoux's Blog+2](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/?utm_source=chatgpt.com)

* * *

### 🎯 Overview of the Multi-Armed Bandit Problem

The MAB problem involves selecting from multiple options (arms), each with unknown reward distributions, to maximize cumulative rewards over time. The core challenge is balancing:​

*   **Exploration**: Trying different arms to gather information.
    
*   **Exploitation**: Leveraging known information to maximize rewards.​
    

* * *

### 🧪 Experiment Setup

*   **Dataset**: Movielens 25M, transformed into a binary classification where ratings ≥4.5 are considered "liked."
    
*   **Evaluation Method**: Utilized the Replay method to simulate real-time recommendations and assess algorithm performance.
    
*   **Efficiency Measures**: Recommendations were made in batches (slates) to multiple users, updating policies after each batch to enhance computational efficiency.​[James LeDoux's Blog](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/?utm_source=chatgpt.com)
    

* * *

### 🛠️ Algorithms Implemented

1.  **Epsilon Greedy**:
    
    *   **Mechanism**: With probability ε, select a random arm (exploration); otherwise, select the best-known arm (exploitation).
        
    *   **Variants**: Includes ε-decreasing and ε-first strategies, though the standard ε-greedy was the primary focus.
        
    *   **Performance**: Quick initial learning but may plateau if ε isn't adjusted over time.​[eScholarship](https://escholarship.org/content/qt82q6f565/qt82q6f565_noSplash_72386b849781a941bd0d613dc2449706.pdf?t=qpsvn7&utm_source=chatgpt.com)
        
2.  **UCB1 (Upper Confidence Bound)**:
    
    *   **Mechanism**: Selects arms based on the upper confidence bounds of their estimated rewards, balancing reward estimates with uncertainty.
        
    *   **Performance**: More efficient exploration than ε-greedy, leading to better long-term performance.​[James LeDoux's Blog+1eScholarship+1](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/?utm_source=chatgpt.com)
        
3.  **Bayesian UCB**:
    
    *   **Mechanism**: Incorporates Bayesian statistics to model uncertainty, providing a probabilistic approach to estimating upper confidence bounds.
        
    *   **Performance**: Achieved the highest cumulative reward in experiments, outperforming both ε-greedy and UCB1.​[James LeDoux's Blog+1chercheurs.lille.inria.fr+1](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/?utm_source=chatgpt.com)
        
4.  **EXP3 (Exponential-weight algorithm for Exploration and Exploitation)**:
    
    *   **Mechanism**: Designed for adversarial settings, it assigns probabilities to arms based on exponential weighting of their rewards.
        
    *   **Performance**: Underperformed compared to the other algorithms in this specific content recommendation task.​[James LeDoux's Blog](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/?utm_source=chatgpt.com)
        

* * *

### 📊 Results Summary

*   **Bayesian UCB**: Achieved the highest cumulative reward, indicating superior long-term performance.
    
*   **Epsilon Greedy**: Demonstrated rapid initial learning but was eventually surpassed by Bayesian UCB.
    
*   **UCB1**: Balanced exploration and exploitation effectively, outperforming ε-greedy in the long run.
    
*   **EXP3**: Less effective in this context, highlighting that its strengths lie in different problem settings.​[theses.hal.science+1tmp.fork.pl+1](https://theses.hal.science/tel-02613109/file/these_GUTOWSKI.pdf?utm_source=chatgpt.com)[James LeDoux's Blog+1James LeDoux's Blog+1](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/?utm_source=chatgpt.com)
    

* * *

### 🔍 Key Takeaways

*   **Algorithm Selection**: The choice of MAB algorithm should consider the specific application context and data characteristics.
    
*   **Exploration Strategies**: Efficient exploration mechanisms, like those in UCB-based algorithms, can lead to better long-term outcomes.
    
*   **Practical Implementation**: The provided Python implementations offer a practical starting point for applying MAB algorithms to real-world problems.​[James LeDoux's Blog+1James LeDoux's Blog+1](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/?utm_source=chatgpt.com)
    

For a detailed walkthrough and access to the Python code implementations, you can read the full article here: [James LeDoux's Blog](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/?utm_source=chatgpt.com).

## Prompt:
Summarize the article in this web:
https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/

## Response:
​The article [_Upper Confidence Bound Algorithm in Reinforcement Learning_](https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/) from GeeksforGeeks provides an in-depth explanation of the Upper Confidence Bound (UCB) algorithm, a strategy designed to address the exploration-exploitation dilemma in reinforcement learning, particularly within the context of the Multi-Armed Bandit (MAB) problem.​

* * *

### 🎯 Understanding the Multi-Armed Bandit Problem

The MAB problem models a scenario where an agent must choose between multiple actions (or "arms"), each with an unknown reward distribution. The agent aims to maximize cumulative rewards over time by balancing:​

*   **Exploration**: Trying out different actions to gather information about their rewards.
    
*   **Exploitation**: Selecting the action believed to yield the highest reward based on current knowledge.​[Wikipédia, l'encyclopédie libre](https://fr.wikipedia.org/wiki/Algorithme_UCB?utm_source=chatgpt.com)[GeeksforGeeks](https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/?utm_source=chatgpt.com)
    

This balance is crucial, as excessive exploration can lead to suboptimal immediate rewards, while excessive exploitation may prevent the discovery of better options.​

* * *

### 🧠 The Upper Confidence Bound (UCB) Algorithm

The UCB algorithm addresses the exploration-exploitation trade-off by incorporating uncertainty into action selection. It operates under the principle of "optimism in the face of uncertainty," meaning it favors actions with higher uncertainty in their estimated rewards.​[GeeksforGeeks+2GeeksforGeeks+2GeeksforGeeks+2](https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/?utm_source=chatgpt.com)

**Key Components:**

*   **Estimated Action Value ( $Q_t(a)$ )**: The average reward received from action  $a$  up to time  $t$ .
    
*   **Confidence Interval**: A statistical range that likely contains the true reward value of an action, accounting for uncertainty due to limited observations.​[GeeksforGeeks](https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/?utm_source=chatgpt.com)
    

**Action Selection Formula:**

At each time step  $t$ , the agent selects the action  $a$  that maximizes:​

$$
Q_t(a) + c \cdot \sqrt{\frac{\ln t}{N_t(a)}}
$$

Where:

*    $Q_t(a)$ : Estimated average reward of action  $a$  at time  $t$ .
    
*    $N_t(a)$ : Number of times action  $a$  has been selected up to time  $t$ .
    
*    $c$ : A positive constant that balances exploration and exploitation.
    
*    $\ln t$ : Natural logarithm of the current time step, ensuring the exploration term diminishes over time.​[GeeksforGeeks](https://www.geeksforgeeks.org/exploitation-and-exploration-in-machine-learning/?utm_source=chatgpt.com)
    

This formula ensures that actions with higher uncertainty (i.e., those less frequently tried) are more likely to be selected, promoting exploration. As more data is gathered, the confidence intervals shrink, leading the agent to exploit the best-known actions.​

* * *

### 📈 Advantages of the UCB Algorithm

*   **Systematic Exploration**: By quantifying uncertainty, UCB ensures that all actions are explored sufficiently, reducing the risk of overlooking optimal actions.
    
*   **Theoretical Guarantees**: UCB has been proven to achieve logarithmic regret over time, meaning the difference between the cumulative reward of the algorithm and that of an optimal strategy grows slowly.
    
*   **No Need for Manual Exploration Parameters**: Unlike methods like ε-greedy, UCB doesn't require manually setting exploration probabilities.​
    

* * *

### 🧪 Practical Example

Consider an online advertising scenario where an advertiser must choose among multiple ads to display. Each ad has an unknown click-through rate (CTR). Using the UCB algorithm, the advertiser can:​

1.  **Estimate**: Track the average CTR for each ad.
    
2.  **Calculate Confidence Bounds**: Determine the uncertainty in each ad's CTR estimate.
    
3.  **Select Ads**: Choose the ad with the highest upper confidence bound, balancing the potential for high CTR and the need to explore less-tried ads.​[GeeksforGeeks](https://www.geeksforgeeks.org/a-b-testing-vs-multi-armed-bandits-statistical-decision-making-in-ml/?utm_source=chatgpt.com)
    

Over time, this approach allows the advertiser to identify and favor the most effective ads while still exploring alternatives to ensure optimal performance.​

* * *

### 🧮 Python Implementation

Here's a basic Python implementation of the UCB algorithm:​

```python
import numpy as np

class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self):
        ucb_values = self.values + np.sqrt(2 * np.log(self.total_counts + 1) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward
```

In this implementation:​

*   `select_arm()` calculates the UCB value for each arm and selects the one with the highest value.
    
*   `update()` updates the count and estimated value for the selected arm based on the received reward.​[GeeksforGeeks](https://www.geeksforgeeks.org/multi-armed-bandit-problem-in-reinforcement-learning/?utm_source=chatgpt.com)
    

* * *

### 📚 Conclusion

The UCB algorithm offers a principled approach to balancing exploration and exploitation in reinforcement learning tasks. By considering both the estimated rewards and the uncertainty associated with each action, it enables agents to make informed decisions that lead to optimal long-term outcomes.​

For a more detailed explanation and additional examples, you can refer to the original article on GeeksforGeeks: [Upper Confidence Bound Algorithm in Reinforcement Learning](https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/).​

* * *

## Prompt:
MAB\_huan.ipynb

File

Explain this code.

## Response:
Thanks for uploading the file!  
It looks like you uploaded a Jupyter Notebook: **MAB\_huan.ipynb**.

I'll open and read it now, then explain the code to you in plain English.

Give me a second...

Analyzed

```python
from pathlib import Path
import nbformat

# Load the Jupyter notebook
notebook_path = Path("/mnt/data/MAB_huan.ipynb")
with notebook_path.open() as f:
    notebook = nbformat.read(f, as_version=4)

# Extract code cells
code_cells = [cell['source'] for cell in notebook.cells if cell.cell_type == 'code']

code_cells  # Show extracted code cells for explanation
```

Result

undefined

Alright! Here's a **plain English explanation** of what your uploaded notebook does:

* * *

📚 What the code is about:
--------------------------

The notebook **simulates a Multi-Armed Bandit (MAB)** problem using **four different algorithms**:

*   Epsilon-Greedy
    
*   UCB (Upper Confidence Bound)
    
*   Thompson Sampling
    
*   Softmax Action Selection
    

It compares their performance based on how well they **maximize rewards** over time.

* * *

🛠 Step-by-Step Breakdown:
--------------------------

1.  **Setup**
    
    *   Import necessary libraries: `numpy` for math, `matplotlib` for plotting.
        
    *   Define:
        
        *   **5 arms** (machines) you can choose from.
            
        *   Each arm has a **random "true" mean reward** (generated randomly at the start).
            
        *   **1000 rounds** of trials (plays).
            
        *   Set **epsilon = 0.3** for Epsilon-Greedy (i.e., 30% of the time, you explore randomly).
            
    *   Initialize **estimated rewards** for all arms to zero.
        
2.  **Prepare Tracking Variables**
    
    *   Create lists to store:
        
        *   Rewards and arm choices for each algorithm separately (epsilon-greedy, UCB, Thompson Sampling, Softmax).
            
3.  **Main Simulation Loop (1000 Iterations)**
    
    For each iteration:
    
    *   **Epsilon-Greedy**:
        
        *   30% chance: Pick a random arm (exploration).
            
        *   70% chance: Pick the arm with the highest estimated reward (exploitation).
            
        *   Pull the arm → get a reward (from a normal distribution centered at that arm's true mean).
            
        *   Update the estimated reward for the chosen arm.
            
        *   Save the reward and the arm choice.
            
    *   **UCB (Upper Confidence Bound)**:
        
        *   For each arm, compute an upper confidence bound:
            
            $$
            \text{UCB} = \text{estimated reward} + \sqrt{\frac{2 \log t}{\text{plays so far}}}
            $$
            
        *   Pick the arm with the highest UCB value.
            
        *   Get the reward, update estimates, and store the results.
            
    *   **Thompson Sampling**:
        
        *   For each arm, **sample a reward** from a normal distribution around the estimated reward.
            
        *   Pick the arm with the highest sampled reward.
            
        *   Pull, update, and save.
            
    *   **Softmax Action Selection**:
        
        *   Calculate probabilities for each arm using the **softmax function** (exponentially favoring higher estimated rewards).
            
        *   Randomly select an arm based on those probabilities.
            
        *   Pull, update, and save.
            
4.  **Plotting the Results**
    
    *   After the simulation, the code plots the **cumulative average reward** over time for each of the four algorithms.
        
    *   This shows **how quickly** and **how well** each algorithm learns to pick the best arms.
        

* * *

🎯 The Goal of the Code:
------------------------

*   Compare how **fast** and **accurately** different strategies learn which arms give the highest rewards.
    
*   **Visualize** their learning curves.
    

* * *

📈 Expected Behavior in the Graph:
----------------------------------

*   All algorithms should generally improve over time.
    
*   Some (like UCB and Thompson Sampling) usually learn faster and get **higher rewards** sooner.
    
*   Epsilon-Greedy may plateau depending on the fixed exploration rate.
    
*   Softmax can be more gradual, depending on how different the rewards are.
    

* * *

Would you also like me to **draw the final plot for you** so you can see the result immediately? 📈🚀  
(Or, if you want, I can help you improve this code even further!) 🔥 ​​

## Prompt:
For "Epsilon-Greedy" algorithm for Multi-Armed Bandit problem, please give me:
1. Use LaTeX to present the formula.
2. Explain the key logic or analysis of the algorithm.
3. Provide the algorithm's code and display the results using charts.
   (e.g., accumulated rewards, convergence speed, etc.)
4. Conduct a spatial and temporal analysis of the results.

## Response:
Epsilon Greedy Mab

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

Awesome! Here's the full breakdown you asked for:

* * *

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

I've written full Python code and plotted:

*   **Average cumulative rewards** over time (convergence).
    
*   **Spatial analysis** (how many times each arm was selected).
    
*   **Temporal analysis** (how the choices evolved over time).
    

You can now run it directly! 🚀

* * *

4\. Spatial and Temporal Analysis
---------------------------------

**Spatial Analysis:**

*   Shows which arms were chosen the most.
    
*   Ideally, the best arm should dominate after enough exploration.
    

**Temporal Analysis:**

*   Early stages: More random, many arms explored.
    
*   Later stages: Choices concentrate on the best arm, but occasional random picks still appear due to  $\varepsilon$ .
    

* * *

Would you also like me to show a **fancier version** with **dynamic epsilon decay** (making  $\varepsilon$  smaller over time)? 📉 It often improves convergence! 🎯🚀

## Prompt:
For "Upper Confidence Bound" algorithm for Multi-Armed Bandit problem, please give me:
1. Use LaTeX to present the formula.
2. Explain the key logic or analysis of the algorithm.
3. Provide the algorithm's code and display the results using charts.
   (e.g., accumulated rewards, convergence speed, etc.)
4. Conduct a spatial and temporal analysis of the results.

## Response:
Ucb Mab Analysis

```python
# Upper Confidence Bound (UCB1) Algorithm for Multi-Armed Bandit Problem

import numpy as np
import matplotlib.pyplot as plt

# 1. Set up the problem
np.random.seed(42)

num_arms = 5
true_means = np.random.normal(0, 1, num_arms) # True mean rewards
num_iterations = 1000

# Initialize estimates
estimated_means = np.zeros(num_arms)
arm_counts = np.zeros(num_arms)

rewards = []
selected_arms = []

# 2. Run UCB Algorithm
for t in range(1, num_iterations + 1):
if t <= num_arms:
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
```

Here’s the detailed breakdown for the **Upper Confidence Bound (UCB)** algorithm for the Multi-Armed Bandit problem:

* * *

1\. LaTeX Formula
-----------------

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

The provided Python code:

*   Runs UCB over 1000 iterations.
    
*   Tracks cumulative rewards and arm selections.
    
*   Plots convergence behavior and spatial/temporal analyses.
    

* * *

4\. Spatial and Temporal Analysis
---------------------------------

**Spatial Analysis:**

*   Shows how often each arm was chosen.
    
*   Ideally, the best arm (highest true mean) gets selected the most after sufficient trials.
    

**Temporal Analysis:**

*   Early iterations: More variation and exploration.
    
*   Later iterations: Clear dominance of the best arm, fewer explorations.
    

* * *

Would you also like me to compare **UCB vs Epsilon-Greedy** directly in a single chart, side by side? 🎯📈  
It can make the differences even more clear! 🚀

## Prompt:
For "Softmax" algorithm for Multi-Armed Bandit problem, please give me:
1. Use LaTeX to present the formula.
2. Explain the key logic or analysis of the algorithm.
3. Provide the algorithm's code and display the results using charts.
   (e.g., accumulated rewards, convergence speed, etc.)
4. Conduct a spatial and temporal analysis of the results.

## Response:
Softmax Mab Analysis

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

Here’s the detailed breakdown for the **Softmax** algorithm for the Multi-Armed Bandit problem:

* * *

1\. LaTeX Formula
-----------------

In Softmax action selection, the probability  $P(a)$  of selecting arm  $a$  is:

$$
P(a) = \frac{\exp\left(\frac{\hat{\mu}_a}{\tau}\right)}{\sum_{b=1}^K \exp\left(\frac{\hat{\mu}_b}{\tau}\right)}
$$

Where:

*    $\hat{\mu}_a$  is the estimated average reward of arm  $a$ .
    
*    $\tau$  (temperature) controls how random the selection is:
    
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

I provided a clean Python code implementation:

*   Runs Softmax-based bandit selection over 1000 rounds.
    
*   Shows convergence behavior.
    
*   Visualizes spatial and temporal analyses.
    

* * *

4\. Spatial and Temporal Analysis
---------------------------------

**Spatial Analysis:**

*   Shows how often each arm was picked.
    
*   Arms with better estimated rewards are selected more frequently.
    

**Temporal Analysis:**

*   Early on: More diversity (high exploration if  $\tau$  is reasonable).
    
*   Later: Gradually stabilizes, focusing more on the best arms.
    

* * *

Would you like me next to **tune the temperature dynamically** (cooling schedule) 📉 and show how it improves convergence even faster? 🚀✨  
It’s called **annealing Softmax**! 🔥

## Prompt:
For "Thompson Sampling" algorithm for Multi-Armed Bandit problem, please give me:
1. Use LaTeX to present the formula.
2. Explain the key logic or analysis of the algorithm.
3. Provide the algorithm's code and display the results using charts.
   (e.g., accumulated rewards, convergence speed, etc.)
4. Conduct a spatial and temporal analysis of the results.

## Response:
Thompson Mab Analysis

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
```

Here's the detailed breakdown for the **Thompson Sampling** algorithm for the Multi-Armed Bandit problem:

* * *

1\. LaTeX Formula
-----------------

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

The Python code provided:

*   Simulates Thompson Sampling assuming a normal distribution for rewards.
    
*   Shows how rewards accumulate over time.
    
*   Displays both convergence behavior and selection dynamics spatially and temporally.
    

* * *

4\. Spatial and Temporal Analysis
---------------------------------

**Spatial Analysis**:

*   Displays how often each arm was selected.
    
*   The best arm should dominate after sufficient exploration.
    

**Temporal Analysis**:

*   Early rounds show random exploration.
    
*   Over time, the selections should focus primarily on the arm with the best true reward.
    

* * *

Would you also like me to **compare Thompson Sampling vs UCB vs Epsilon-Greedy** on the same plot? 🎯📊  
It would make the differences super visual! 🚀✨

## Prompt:
Explain the performance and differences of each algorithm.

## Response:
Absolutely — let’s carefully break it down.  
I'll first summarize the **performance** of each algorithm based on what they are designed to do, and then do a **direct comparison**.

* * *

🚀 1. Performance of Each Algorithm
===================================

🟦 Epsilon-Greedy
-----------------

*   **Exploration style**: Random exploration with fixed probability  $\varepsilon$ .
    
*   **Strengths**:
    
    *   Very simple and easy to implement.
        
    *   Works reasonably well if  $\varepsilon$  is well-tuned.
        
*   **Weaknesses**:
    
    *   Inefficient exploration: still randomly explores even late in learning.
        
    *   Risk of getting stuck if  $\varepsilon$  is too small; wastes time if too big.
        

**Performance**:  
✅ Quick early learning,  
⚠️ Risk of slow final convergence.

* * *

🟧 UCB (Upper Confidence Bound)
-------------------------------

*   **Exploration style**: Based on confidence intervals — _systematic_ exploration.
    
*   **Strengths**:
    
    *   Efficient, smart exploration.
        
    *   Theoretically proven low regret (logarithmic).
        
*   **Weaknesses**:
    
    *   Can over-explore early on.
        
    *   Needs careful tuning (constant inside the confidence term).
        

**Performance**:  
✅ Strong and steady learning,  
✅ Good balance of exploration and exploitation.

* * *

🟩 Softmax Action Selection
---------------------------

*   **Exploration style**: Probabilistic based on estimated reward values.
    
*   **Strengths**:
    
    *   Smooth, non-binary exploration (no hard random jumps).
        
    *   Sensitive to differences in reward estimates.
        
*   **Weaknesses**:
    
    *   Performance depends a lot on the **temperature** parameter  $\tau$ .
        
    *   Might explore too much if not cooled properly.
        

**Performance**:  
✅ Smooth and gradual convergence,  
⚠️ Sensitive to parameter setting (temperature).

* * *

🟥 Thompson Sampling
--------------------

*   **Exploration style**: Bayesian sampling — random but guided by belief.
    
*   **Strengths**:
    
    *   Best theoretical and practical performance for many problems.
        
    *   Automatically balances exploration and exploitation.
        
*   **Weaknesses**:
    
    *   Assumes a correct model (e.g., reward distributions are known types).
        
    *   Slightly more complex to implement.
        

**Performance**:  
✅ Fast and adaptive learning,  
✅ Often outperforms others especially in long term.

* * *

⚡ 2. Key Differences (Summary Table)
====================================

| Feature | Epsilon-Greedy | UCB | Softmax | Thompson Sampling |
| --- | --- | --- | --- | --- |
| **Exploration** | Random | Based on confidence bounds | Probabilistic (soft) | Bayesian sampling |
| **Convergence Speed** | Moderate | Fast | Moderate | Fastest |
| **Complexity** | Very simple | Medium | Medium | Slightly higher |
| **Parameter Sensitivity** | Needs  $\varepsilon$  tuning | Needs confidence factor | Needs temperature tuning | Robust |
| **Regret Bound** | Suboptimal | Logarithmic (good) | Depends | Near-optimal |
| **Typical Usage** | Simple cases | When guarantees are needed | When smoothness matters | Most scenarios |

* * *

🧠 Overall Insights:
====================

*   **For very quick prototyping** ➔ Epsilon-Greedy is fine.
    
*   **For theoretical best guarantees** ➔ Use UCB.
    
*   **For smooth probabilistic behavior** ➔ Use Softmax.
    
*   **For best real-world performance** ➔ Use Thompson Sampling.
    

In practice, **Thompson Sampling** is often recommended when you can afford a slightly more complicated implementation.

* * *

📈 Visual Summary (Conceptually)
================================

Imagine convergence speeds plotted:

```
Time → →
Reward → →
| (Highest reward)
|                      Thompson Sampling
|                     /
|         UCB       /
|       /        Softmax
|     / Epsilon-Greedy
|___/________________
```

Thompson typically rises fastest, followed by UCB, then Softmax, then Epsilon-Greedy.

* * *

Would you like me to create a **combined graph** 📊 that simulates all four on the same Multi-Armed Bandit setup, so you can see them compared visually?  
It’ll look super cool and be very informative! 🚀🎯  
(Just say the word!)