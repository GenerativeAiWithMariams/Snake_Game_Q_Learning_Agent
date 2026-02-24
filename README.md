# 🐍 Snake Q-Learning Agent (Tabular Reinforcement Learning)

## 📌 Project Overview

This project implements a **Snake game using Q-Learning (Tabular Reinforcement Learning)** built with:

* Python
* Pygame
* NumPy

The agent learns how to move toward food and avoid collisions through reward-based learning on a small grid environment.

---

## 🎯 Objective

The goal of the agent is to:

* ✅ Eat food (+10 reward)
* ❌ Avoid collision with walls or itself (-100 reward)
* ➖ Minimize unnecessary movement (-1 step penalty)

The agent improves over multiple episodes using the Q-learning update rule.

---

## 🧠 Reinforcement Learning Setup

### 🔹 State Representation

State is defined as:

```
(head_x, head_y, food_x, food_y)
```

This allows the agent to know:

* Its current position
* The food position

Q-table shape:

```
(GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE, 4)
```

Where:

* 4 = number of possible actions (UP, DOWN, LEFT, RIGHT)

---

### 🔹 Action Space

```
UP
DOWN
LEFT
RIGHT
```

---

### 🔹 Reward Function

| Condition   | Reward |
| ----------- | ------ |
| Eat Food    | +10    |
| Collision   | -100   |
| Normal Move | -1     |

---

### 🔹 Q-Learning Update Formula

[
Q(s,a) = Q(s,a) + \alpha [r + \gamma \max Q(s',a') - Q(s,a)]
]

Where:

* α = Learning rate
* γ = Discount factor
* r = Reward
* s' = Next state

---

## ⚙️ Hyperparameters

```python
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 500
```

* Uses epsilon-greedy exploration
* Epsilon decays over time

---

## 🎮 Game Features

* Dynamic snake growth
* Food never spawns inside snake
* Collision detection
* Grid visualization
* Live episode training
* Epsilon decay
* Tabular Q-learning implementation

---

## 🖥️ Requirements

Install dependencies:

```bash
pip install pygame numpy
```

---

## ▶️ How to Run

```bash
python snake_q_learning.py
```

---

## 📊 Expected Behavior

Early Episodes:

* Random movement
* Frequent collisions
* Negative total reward

Later Episodes:

* Improved movement toward food
* Fewer collisions
* Higher total reward

---

## 🚀 Learning Outcomes

This project demonstrates:

* Reinforcement Learning fundamentals
* Q-learning implementation from scratch
* State design for RL problems
* Epsilon-greedy exploration
* Game environment simulation using Pygame

---

## 🔮 Possible Improvements

* Remove rendering during training for faster learning
* Save and load Q-table
* Increase grid size
* Add score tracking UI
* Upgrade to Deep Q-Network (DQN)
* Implement reward shaping
* Add replay memory

---

## 📁 Project Structure

```
snake_q_learning.py
README.md
```

---

## 👩‍💻 Author

Maryam S
Generative AI & Machine Learning Developer
