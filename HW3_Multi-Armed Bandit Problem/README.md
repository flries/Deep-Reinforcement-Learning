# Performance of Each Algorithm

## 1. Epsilon-Greedy

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


## 2. UCB (Upper Confidence Bound)

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


## 3. Softmax Action Selection

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


## 4. Thompson Sampling

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

# Key Differences (Summary Table)

| Feature | Epsilon-Greedy | UCB | Softmax | Thompson Sampling |
| --- | --- | --- | --- | --- |
| **Exploration** | Random | Based on confidence bounds | Probabilistic (soft) | Bayesian sampling |
| **Convergence Speed** | Moderate | Fast | Moderate | Fastest |
| **Complexity** | Very simple | Medium | Medium | Slightly higher |
| **Parameter Sensitivity** | Needs  $\varepsilon$  tuning | Needs confidence factor | Needs temperature tuning | Robust |
| **Regret Bound** | Suboptimal | Logarithmic (good) | Depends | Near-optimal |
| **Typical Usage** | Simple cases | When guarantees are needed | When smoothness matters | Most scenarios |

* * *

# Overall Insights:

*   **For very quick prototyping** ➔ Epsilon-Greedy is fine.
*   **For theoretical best guarantees** ➔ Use UCB.
*   **For smooth probabilistic behavior** ➔ Use Softmax.
*   **For best real-world performance** ➔ Use Thompson Sampling.
    

In practice, **Thompson Sampling** is often recommended when you can afford a slightly more complicated implementation.