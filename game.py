import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN1 = (66, 245, 102)
GREEN2 = (40,158,63)

BLOCK_SIZE = 20
SPEED = 30


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        self.APPLE = pygame.image.load("red-apple.png")
        self.APPLE = pygame.transform.scale(self.APPLE,(BLOCK_SIZE, BLOCK_SIZE))
        self.HEAD = pygame.image.load("head.png")
        self.HEAD = pygame.transform.scale(self.HEAD,(30, 30))

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action, headless):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        reward = 0
        # if not np.array_equal(action, [1, 0, 0]):
        #     reward = -1 #small penalty for making a turn

        # 3. check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -30
            if self.is_snake_collision():
                reward = -40
            return reward, game_over, self.score

        if self.frame_iteration > 100*len(self.snake):
            game_over = True
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 30
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        if not headless:
            self._update_ui()
            self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_snake_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        return self.is_snake_collision(pt)

    def _update_ui(self):
        self.display.fill(BLACK)

        pt = self.snake[0]
        if self.direction == Direction.RIGHT:
            head_img = pygame.transform.rotate(self.HEAD,-90)
        elif self.direction == Direction.LEFT:
            head_img = pygame.transform.rotate(self.HEAD,90)
        elif self.direction == Direction.DOWN:
            head_img = pygame.transform.rotate(self.HEAD,180)
        else:
            head_img = self.HEAD

        self.display.blit(head_img,(pt.x-5,pt.y-5))

        for pt in self.snake[1:]:
            pygame.draw.circle(self.display, color=GREEN1, center=(pt.x+BLOCK_SIZE/2, pt.y+BLOCK_SIZE/2), radius=BLOCK_SIZE/2 )
            pygame.draw.circle(self.display, color=GREEN2, center=(pt.x+BLOCK_SIZE/2, pt.y+BLOCK_SIZE/2), radius=BLOCK_SIZE/3, width=3 )

        # pygame.draw.rect(self.display, RED, pygame.Rect(
        #     self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        self.display.blit(self.APPLE, (self.food.x, self.food.y))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
