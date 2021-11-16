import random
import numpy as np
import time
import itertools
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from enum import Enum

class TurnDirection(Enum):
    RIGHT = 1
    LEFT = 2
    STRAIGHT = 3


MAX_MEMORY = 100_000
BATCH_SIZE = 10000
LR = 0.001

class Agent:

    def __init__(self, reload):
        self.n_games = 0
        self.epsilon = 160 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3, lr=LR)
        self.trainer = QTrainer(self.model, gamma=self.gamma)

        if reload:
            print("Loaded")
            self.model.load(reload)
            self.epsilon = 5

    def _new_dir(self, current_direction : Direction, turn_direction : TurnDirection):
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                Direction.LEFT, Direction.UP]
        idx = clock_wise.index(current_direction)

        if turn_direction == TurnDirection.STRAIGHT:
            new_dir = current_direction  # no change
        elif turn_direction == TurnDirection.RIGHT:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # LEFT
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

    def _get_snake_collision_danger(self, game, travel_direction : Direction, turn_direction : TurnDirection, head, offset):
        dir_l = travel_direction == Direction.LEFT
        dir_r = travel_direction == Direction.RIGHT
        dir_u = travel_direction == Direction.UP
        dir_d = travel_direction == Direction.DOWN

        point_l = Point(head.x - offset, head.y)
        point_r = Point(head.x + offset, head.y)
        point_u = Point(head.x, head.y - offset)
        point_d = Point(head.x, head.y + offset)

        rating = 0

        if turn_direction == TurnDirection.STRAIGHT:
            rating +=(
                    (dir_r and game.is_snake_collision(point_r)) or 
                    (dir_l and game.is_snake_collision(point_l)) or 
                    (dir_u and game.is_snake_collision(point_u)) or 
                    (dir_d and game.is_snake_collision(point_d)))

        elif turn_direction == TurnDirection.RIGHT:
            rating +=(
                    (dir_u and game.is_snake_collision(point_r)) or 
                    (dir_d and game.is_snake_collision(point_l)) or 
                    (dir_l and game.is_snake_collision(point_u)) or 
                    (dir_r and game.is_snake_collision(point_d)))
        else:
            # LEFT
            rating +=(
                    (dir_d and game.is_snake_collision(point_r)) or 
                    (dir_u and game.is_snake_collision(point_l)) or 
                    (dir_r and game.is_snake_collision(point_u)) or 
                    (dir_l and game.is_snake_collision(point_d)))

        return rating

    def _get_danger(self, game, turn_direction : TurnDirection):
        LOOK_AHEAD = 5

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        head = game.snake[0]


        rating = 0
        offset = 20
        point_l = Point(head.x - offset, head.y)
        point_r = Point(head.x + offset, head.y)
        point_u = Point(head.x, head.y - offset)
        point_d = Point(head.x, head.y + offset)

        # Look for collision danger for next move, include walls as well as bits of snake
        if turn_direction == TurnDirection.STRAIGHT:
            rating +=(
                    (dir_r and game.is_collision(point_r)) or 
                    (dir_l and game.is_collision(point_l)) or 
                    (dir_u and game.is_collision(point_u)) or 
                    (dir_d and game.is_collision(point_d)))

        elif turn_direction == TurnDirection.RIGHT:
            rating +=(
                    (dir_u and game.is_collision(point_r)) or 
                    (dir_d and game.is_collision(point_l)) or 
                    (dir_l and game.is_collision(point_u)) or 
                    (dir_r and game.is_collision(point_d)))
        else:
            # LEFT
            rating +=(
                    (dir_d and game.is_collision(point_r)) or 
                    (dir_u and game.is_collision(point_l)) or 
                    (dir_r and game.is_collision(point_u)) or 
                    (dir_l and game.is_collision(point_d)))

        if rating:
            return rating

        ## ok, no immediate collision danger, now look ahead, but only for bits of snake, ignore walls
        for i in range(1, LOOK_AHEAD):
            offset = 20 * (i+1)
            rating += self._get_snake_collision_danger(game, game.direction, turn_direction, head, offset) * (LOOK_AHEAD - i)
            if rating:
                break

        return rating / LOOK_AHEAD


       

    def get_state(self, game):
        head = game.snake[0]
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            self._get_danger(game, TurnDirection.STRAIGHT),

            # Danger right
            self._get_danger(game, TurnDirection.RIGHT),

            # Danger left
            self._get_danger(game, TurnDirection.LEFT),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            (game.food.x < game.head.x),  # food left
            (game.food.x > game.head.x),  # food right
            (game.food.y < game.head.y),  # food up
            (game.food.y > game.head.y)   # food down
            ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self, number_of_steps):
        if len(self.memory) - number_of_steps > BATCH_SIZE:
            local_memory = list(self.memory)
            short_memory = local_memory[-number_of_steps:]
            mini_sample = random.sample(local_memory[:-number_of_steps], BATCH_SIZE) # list of tuples
            mini_sample.extend(short_memory)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        #self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = np.array(state, dtype=np.float)[np.newaxis,...]
            prediction = self.model(state0)
            move = np.argmax(prediction, axis=1)[0]
            final_move[move] = 1

        return final_move