# HW1-2_Policy Display and Value Evaluation

## Project Introduction
Assigning a random allowed action (displayed as an arrow) to each cell (excluding start, end, and obstacle cells) and computing a state value matrix using a policy evaluation algorithm based on value iteration.

## Target Function
**Policy Matrix Generation:**    
  Each cell is assigned a random allowed arrow (policy) considering grid boundaries and the position of start/end cells.    

**Value Evaluation:**     
  Computes cell values using value iteration and policy evaluation with the following reward system:    
  - *Normal cells:* -1
  - *End cell:* +10
  - *Obstacle cell:* -6

  Uses a policy evaluation equation:    
  $V(s) = R(s) + \gamma \sum P(s' \mid s,a) V(s')$    
  - *R(s):* Reward for the current cell.
  - *γ (gamma):* Discount factor (set to 0.9).
  - *Transition Probabilities:* 0.8 for intended direction, 0.1 for perpendicular movements.

## Environment Requirements
- **Programming Language:**    
`Python 3.x`    

- **Web Framework:**    
`Flask` *(tested with version 2.x)*    

- **Frontend Dependencies:**    
`HTML`, `CSS`, `JavaScript` *(jQuery is included via CDN)*    

- **Web Browser:**    
Any modern web browser *(Chrome, Firefox, Safari, etc.)*

## Installation and Setup:
1. **Clone or Download the Project:**    
   Clone the repository:    
   ```bash
   git clone <repository_url>
   ```
   Or download the project files directly.

2. **Install Required Packages:**    
   Ensure you have Python 3 installed.    
   Install Flask using pip:
   ```bash
   pip install flask
   ```

3. **Run the Application:**    
   Navigate to the project directory in your terminal.    
   Start the Flask server by running:
   ```python
   python app.py
   ```
   The server will start in debug mode and listen on the default port (5000).

4. **Access the Application:**    
   Open your web browser and navigate to http://127.0.0.1:5000/

## Program Result
![HW1-2_Result_01](https://github.com/user-attachments/assets/cab1517a-6f5c-4220-bb65-0f62d0675169)
![HW1-2_Result_02](https://github.com/user-attachments/assets/2120b270-7740-416a-8635-4d9453f935c6)
![HW1-2_Result_03](https://github.com/user-attachments/assets/3bb213d6-f9f5-479c-89a6-86253ca04af6)
