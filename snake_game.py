import pygame
import random
import numpy as np

pygame.init()
# ===============================
# SETTINGS
# ===============================
WIDTH = 500
HEIGHT = 500
GRID_SIZE = 5
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 5

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0 ,0)
BLACK = (0, 0 ,0)

# ===============================
# ACTIONS
# ===============================
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# ===============================
# WINDOW
# ===============================
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Q-learning - Small Grid")
font = pygame.font.SysFont(None, 24)

# ===============================
# FOOD CLASS
# ===============================
class Food:
    def __init__(self, snake_positions):
        self.randomize(snake_positions)

    def randomize(self, snake_positions):
        while True:
            pos = (random.randint(0, GRID_SIZE - 1),
                   random.randint(0, GRID_SIZE - 1))
            if pos not in snake_positions:
                self.position = pos
                break

# ===============================
# SNAKE CLASS
# ===============================
class Snake:
    def __init__(self):
        self.positions = [(2, 2)]
        self.direction = random.choice(ACTIONS)

    def move(self, action, grow=False):
        self.direction = action
        head_x, head_y = self.positions[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)

        if grow:
            self.positions = [new_head] + self.positions
        else:
            self.positions = [new_head] + self.positions[:-1]

    def collision(self):
        head = self.positions[0]
        return (
            head in self.positions[1:] or
            head[0] < 0 or head[0] >= GRID_SIZE or
            head[1] < 0 or head[1] >= GRID_SIZE
        )

# ===============================
# Q-TABLE
# State = (head_x, head_y, food_x, food_y)
# ===============================
q_table = np.zeros((GRID_SIZE, GRID_SIZE,
                    GRID_SIZE, GRID_SIZE, 4))

# ===============================
# HYPERPARAMETERS
# ===============================
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 500

# ===============================
# STATE FUNCTION
# ===============================
def get_state(snake, food):
    head_x, head_y = snake.positions[0]
    food_x, food_y = food.position
    return head_x, head_y, food_x, food_y

# ===============================
# REWARD FUNCTION
# ===============================
def get_reward(snake, food):
    if snake.collision():
        return -100
    elif snake.positions[0] == food.position:
        return 10
    else:
        return -1

# ===============================
# TRAINING LOOP
# ===============================
clock = pygame.time.Clock()

for episode in range(1, num_episodes + 1):

    snake = Snake()
    food = Food(snake.positions)
    done = False
    total_reward = 0

    while not done:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        current_state = get_state(snake, food)

        # Epsilon-Greedy
        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = np.argmax(
                q_table[current_state[0],
                        current_state[1],
                        current_state[2],
                        current_state[3]]
            )

        action = ACTIONS[action_idx]

        # Check if snake will eat
        next_head = (
            snake.positions[0][0] + action[0],
            snake.positions[0][1] + action[1]
        )

        will_grow = (next_head == food.position)

        snake.move(action, grow=will_grow)

        reward = get_reward(snake, food)
        total_reward += reward

        next_state = get_state(snake, food)

        # Q-learning update
        if snake.collision():
            target = reward
            done = True
        else:
            target = reward + gamma * np.max(
                q_table[next_state[0],
                        next_state[1],
                        next_state[2],
                        next_state[3]]
            )

        q_table[current_state[0],
                current_state[1],
                current_state[2],
                current_state[3],
                action_idx] += \
            alpha * (
                target -
                q_table[current_state[0],
                        current_state[1],
                        current_state[2],
                        current_state[3],
                        action_idx]
            )

        # If ate food → spawn new
        if will_grow:
            food.randomize(snake.positions)

        # ===============================
        # DRAWING
        # ===============================
        win.fill(WHITE)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pygame.draw.rect(
                    win, BLACK,
                    (i * CELL_SIZE,
                     j * CELL_SIZE,
                     CELL_SIZE,
                     CELL_SIZE), 1
        )
        # Draw Snake
        for index, pos in enumerate(snake.positions):
            pygame.draw.rect(
                win, GREEN,
                (pos[0] * CELL_SIZE,
                 pos[1] * CELL_SIZE,
                 CELL_SIZE,
                 CELL_SIZE)
    )
            # Show label only on head
            if index == 0:
                label = font.render("Snake", True, BLACK)
                win.blit(label,
                         (pos[0] * CELL_SIZE + 5,
                          pos[1] * CELL_SIZE + 5)
    )   
        
        # Draw Food
        pygame.draw.rect(
            win, RED,
            (food.position[0] * CELL_SIZE,
             food.position[1] * CELL_SIZE,
             CELL_SIZE,
             CELL_SIZE)
)

        food_label = font.render("Food", True, BLACK)
        win.blit(food_label,
                 (food.position[0] * CELL_SIZE + 5,
                  food.position[1] * CELL_SIZE + 5))
        pygame.display.update()

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode} | Total Reward: {total_reward} | Epsilon: {round(epsilon, 3)}")


pygame.quit()
