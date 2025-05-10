import torch 
import random
import numpy as np
from collections import deque
from game import SnakeEnvironment, MovementDirection, Coordinate 
from model import DQNetwork, DQNTrainer
from visualization import display_metrics

# Configuration parameters
MEMORY_CAPACITY = 120_000
TRAINING_BATCH = 1200
LEARNING_RATE = 0.0008
EXPLORATION_DECAY = 0.6  

class SnakeAgent:
    """Agent that learns to play Snake using Deep Q-Learning"""

    def __init__(self):
        self.games_played = 0
        self.exploration_rate = 0.9 
        self.discount_factor = 0.85  
        self.experience_buffer = deque(maxlen=MEMORY_CAPACITY)
        self.neural_net = DQNetwork(11, [128, 64], 3)
        self.trainer = DQNTrainer(self.neural_net, lr=LEARNING_RATE, discount=self.discount_factor)

    def observe_environment(self, game):
        head = game.snake[0]
        
        
        left_pos = Coordinate(head.x - 20, head.y)
        right_pos = Coordinate(head.x + 20, head.y)
        up_pos = Coordinate(head.x, head.y - 20)
        down_pos = Coordinate(head.x, head.y + 20)
        
        facing_left = game.direction == MovementDirection.LEFT
        facing_right = game.direction == MovementDirection.RIGHT
        facing_up = game.direction == MovementDirection.UP
        facing_down = game.direction == MovementDirection.DOWN

       
        state = [
            (facing_right and game.detect_collision(right_pos)) or 
            (facing_left and game.detect_collision(left_pos)) or 
            (facing_up and game.detect_collision(up_pos)) or 
            (facing_down and game.detect_collision(down_pos)),
 
            (facing_up and game.detect_collision(right_pos)) or 
            (facing_down and game.detect_collision(left_pos)) or 
            (facing_left and game.detect_collision(up_pos)) or 
            (facing_right and game.detect_collision(down_pos)),

            (facing_down and game.detect_collision(right_pos)) or 
            (facing_up and game.detect_collision(left_pos)) or 
            (facing_right and game.detect_collision(up_pos)) or 
            (facing_left and game.detect_collision(down_pos)),
        
            facing_left,
            facing_right,
            facing_up,
            facing_down,
            
            game.food.x < game.head.x,  
            game.food.x > game.head.x,  
            game.food.y < game.head.y,  
            game.food.y > game.head.y   
        ]

        return np.array(state, dtype=int) 

    def store_experience(self, state, action, reward, next_state, done):
        self.experience_buffer.append((state, action, reward, next_state, done))

    def learn_from_experiences(self):
        if len(self.experience_buffer) > TRAINING_BATCH:
            experience_batch = random.sample(self.experience_buffer, TRAINING_BATCH)
        else: 
            experience_batch = self.experience_buffer

        states, actions, rewards, next_states, terminals = zip(*experience_batch)
        self.trainer.optimize(states, actions, rewards, next_states, terminals)

    def quick_update(self, state, action, reward, next_state, done):
        self.trainer.optimize(state, action, reward, next_state, done)

    def select_action(self, state):
        self.exploration_rate = max(5, 100 - self.games_played * EXPLORATION_DECAY) / 100
        
        action = [0, 0, 0]
        
        if random.random() < self.exploration_rate:
            move = random.randint(0, 2)
            action[move] = 1
        # Exploitation
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            q_values = self.neural_net(state_tensor)
            move = torch.argmax(q_values).item()
            action[move] = 1
            
        return action

def run_training():
    game_scores = []
    running_avg_scores = []
    cumulative_score = 0
    best_score = 0 
    
    agent = SnakeAgent()
    environment = SnakeEnvironment()
    
    while True:
        current_state = agent.observe_environment(environment)

        selected_action = agent.select_action(current_state)

        reward, terminal, score = environment.step(selected_action)
        new_state = agent.observe_environment(environment)

        agent.quick_update(current_state, selected_action, reward, new_state, terminal)

        agent.store_experience(current_state, selected_action, reward, new_state, terminal)

        
        if terminal:
            
            environment.reset()
            agent.games_played += 1
            
            
            agent.learn_from_experiences()

            
            if score > best_score:
                best_score = score
                agent.neural_net.save(f'model_{best_score}.pth')

            
            print(f'Episode {agent.games_played} | Score: {score} | Best: {best_score} | Exploration: {agent.exploration_rate:.2f}')

            game_scores.append(score)
            cumulative_score += score
            avg_score = cumulative_score / agent.games_played
            running_avg_scores.append(avg_score)
            
            
            display_metrics(game_scores, running_avg_scores)

if __name__ == '__main__':
    run_training()