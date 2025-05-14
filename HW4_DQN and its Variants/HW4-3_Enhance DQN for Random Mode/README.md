## Enhanced DQN for Gridworld (Random Mode)
### 🎯 Objective

Improve the baseline Deep Q-Network (DQN) model to handle the more complex **'random' mode** of the Gridworld environment, where initial positions vary each episode — increasing state-space complexity and requiring better generalization.

* * *

### 🧠 Enhancements Applied

1.  **🧱 PyTorch Lightning Structure**
    
    *   Modularized training via `LightningModule`.
        
    *   Clean separation of model, optimizer, and training loop.
        
    *   Simplified reproducibility and logging.
        
2.  **🗃 Experience Collection**
    
    *   Created a custom `GridworldDataset` to simulate gameplay episodes and store transitions `(state, action, reward, next_state, done)`.
        
    *   Adapted to the **random mode** for better diversity in training data.
        
3.  **🎲 Exploration Strategy**
    
    *   Applied ε-greedy policy with decaying ε for exploration during experience generation.
        
    *   Starts at `ε=1.0` and decays linearly to `0.1` over episodes.
        
4.  **📈 Training Setup**
    
    *   Batch size: 32
        
    *   Optimizer: Adam with learning rate = `1e-3`
        
    *   Loss: Mean Squared Error (MSE)
        
    *   Discount factor (γ): `0.9`
        
    *   10 training epochs using Lightning `Trainer`.
        

* * *

### 📊 Results Summary

| Metric | Value (Example Run) |
| --- | --- |
| Training Loss | Decreases over epochs |
| Convergence | Faster and smoother |
| Generalization | Improved due to randomness in states |

> Loss plots and performance metrics show more robust learning compared to the static mode due to enhanced state diversity.

* * *

### ✅ Conclusion

The enhanced DQN using PyTorch Lightning effectively adapts to the **random mode** of Gridworld. With modular structure, experience-based training, and richer state variety, the model generalizes better and is easier to scale or extend (e.g., for Double DQN or Prioritized Replay).