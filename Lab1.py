import pygame
import random
from collections import deque

# Initializing Pygame
pygame.init()

# My screen settings
WIDTH, HEIGHT = 500, 500
ROWS, COLS = 20, 20
CELL_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Grid Navigation")

# Create grid
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
obstacles = []

# To randomly add obstacles
for _ in range(60):
    x, y = random.randint(0, COLS - 1), random.randint(0, ROWS - 1)
    grid[y][x] = 1
    obstacles.append((x, y))

# Place agent and target
agent_pos = (0, 0)
target_pos = (COLS - 1, ROWS - 1)
grid[target_pos[1]][target_pos[0]] = 2  # Target marked as 2 for easy identification

# Directions for agent movement
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]


# BFS function to find shortest path
def bfs(start, goal):
    queue = deque([(start, [start])])
    visited = set()
    visited.add(start)

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == goal:
            return path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < COLS and 0 <= ny < ROWS and (nx, ny) not in visited:
                if grid[ny][nx] != 1:  # Check for obstacles
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    visited.add((nx, ny))
    return []


# Main loop
running = True
path = bfs(agent_pos, target_pos)
path_index = 0

while running:
    screen.fill(WHITE)

    # Draw grid
    for y in range(ROWS):
        for x in range(COLS):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (x, y) == agent_pos:
                pygame.draw.rect(screen, BLUE, rect)
            elif (x, y) == target_pos:
                pygame.draw.rect(screen, GREEN, rect)
            elif (x, y) in obstacles:
                pygame.draw.rect(screen, BLACK, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  # Grid lines

    # Move agent along the path
    if path_index < len(path):
        agent_pos = path[path_index]
        path_index += 1

    # Check if agent has reached the target
    if agent_pos == target_pos:
        print("Target reached!")
        running = False

    pygame.display.flip()
    pygame.time.delay(200)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()