import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np 

pygame.init()
try:
    font = pygame.font.Font('arial.ttf', 25)
except:
    font = pygame.font.SysFont('arial', 25)

class MovementDirection(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
# Using namedtuple for coordinates
Coordinate = namedtuple('Coordinate', 'x, y')

# Game color scheme
BACKGROUND = (10, 10, 10)
FOOD_COLOR = (220, 50, 50)
SNAKE_OUTER = (30, 100, 200)
SNAKE_INNER = (30, 150, 230)
TEXT_COLOR = (240, 240, 240)

# Game configuration
GRID_SIZE = 20
GAME_SPEED = 12  # Slightly faster

class SnakeEnvironment:
    
    def __init__(self, width=640, height=480):
        #  Game dimensions
        self.width = width
        self.height = height
        
        # Display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = MovementDirection.RIGHT
        
        self.head = Coordinate(self.width//2, self.height//2)
        self.snake = [
            self.head, 
            Coordinate(self.head.x-GRID_SIZE, self.head.y),
            Coordinate(self.head.x-(2*GRID_SIZE), self.head.y)
        ]
        
        # Game metrics
        self.score = 0
        self.food = None
        self.place_food()
        self.steps_taken = 0
        
        # Distance tracking for rewards
        self.prev_distance_to_food = self.calculate_food_distance()
   
    def calculate_food_distance(self):
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
    def place_food(self):
        x = random.randint(0, (self.width-GRID_SIZE)//GRID_SIZE) * GRID_SIZE 
        y = random.randint(0, (self.height-GRID_SIZE)//GRID_SIZE) * GRID_SIZE
        self.food = Coordinate(x, y)
        
        if self.food in self.snake:
            self.place_food()
        
    def step(self, action):
    
        self.steps_taken += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        self.move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        current_distance = self.calculate_food_distance()
        
        if self.detect_collision() or self.steps_taken > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
            
            if current_distance < self.prev_distance_to_food:
                reward = 0.1
            elif current_distance > self.prev_distance_to_food:
                reward = -0.1
        
        self.prev_distance_to_food = current_distance
        self.render()
        self.clock.tick(GAME_SPEED)
        
        return reward, game_over, self.score
    
    def detect_collision(self, point=None):
        if point is None:
            point = self.head
            
        if (point.x >= self.width or point.x < 0 or 
            point.y >= self.height or point.y < 0):
            return True
            
        if point in self.snake[1:]:
            return True
        
        return False
        
    def render(self):
        
        self.display.fill(BACKGROUND)
        
        for i, segment in enumerate(self.snake):
            color = SNAKE_OUTER if i == 0 else SNAKE_OUTER
            inner_color = SNAKE_INNER if i == 0 else SNAKE_INNER
            
            pygame.draw.rect(self.display, color, 
                            pygame.Rect(segment.x, segment.y, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(self.display, inner_color, 
                            pygame.Rect(segment.x+4, segment.y+4, 12, 12))
            
        pulse = int(10 * np.sin(pygame.time.get_ticks() * 0.01) + 10)
        food_rect = pygame.Rect(self.food.x, self.food.y, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.display, FOOD_COLOR, food_rect)
        pygame.draw.rect(self.display, (FOOD_COLOR[0]+pulse, 
                                        FOOD_COLOR[1]+pulse, 
                                        FOOD_COLOR[2]+pulse), 
                        pygame.Rect(self.food.x+3, self.food.y+3, 14, 14))
        
        score_text = font.render(f"Score: {self.score}", True, TEXT_COLOR)
        steps_text = font.render(f"Steps: {self.steps_taken}", True, TEXT_COLOR)
        self.display.blit(score_text, [10, 10])
        self.display.blit(steps_text, [10, 40])
        
        pygame.display.flip()
        
    def move(self, action):
        clock_wise = [MovementDirection.RIGHT, MovementDirection.DOWN, 
                     MovementDirection.LEFT, MovementDirection.UP]
                     
        current_idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[current_idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (current_idx + 1) % 4
            new_dir = clock_wise[new_idx]
        else: # [0, 0, 1]
            new_idx = (current_idx - 1) % 4
            new_dir = clock_wise[new_idx]
        
        # Update direction
        self.direction = new_dir
        
        # Update head position
        x, y = self.head.x, self.head.y
        if self.direction == MovementDirection.RIGHT:
            x += GRID_SIZE
        elif self.direction == MovementDirection.LEFT:
            x -= GRID_SIZE
        elif self.direction == MovementDirection.DOWN:
            y += GRID_SIZE
        elif self.direction == MovementDirection.UP:
            y -= GRID_SIZE
            
        self.head = Coordinate(x, y)