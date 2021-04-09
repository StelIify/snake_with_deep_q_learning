import torch
import random
import numpy as np
from collections import deque
from ai_snake import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from helper import plot

BLOCK_SIZE = 20
MAX_MEMORY = 3_000_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.8  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)),

            # Danger right
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_right and game.is_collision(point_down)),

            # Danger left
            (direction_down and game.is_collision(point_right)) or
            (direction_up and game.is_collision(point_left)) or
            (direction_right and game.is_collision(point_up)) or
            (direction_left and game.is_collision(point_down)),

            # Move direction
            direction_left,
            direction_right,
            direction_up,
            direction_down,

            # Food location

            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation
        self.epsilon = 80 - self.number_of_games / 10
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            initial_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(initial_state)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_scores = 0
    record = 0
    agent = Agent()
    agent.model.load("model_test2.pth")
    game = SnakeGameAI()
    while True:
        # get the current state
        old_state = agent.get_state(game)

        final_move = agent.get_action(old_state)
        # perform move and get new state
        reward, game_over, score = game.play_step(final_move)

        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, final_move, reward, new_state, game_over)

        agent.remember(old_state, final_move, reward, new_state, game_over)

        if game_over:
            # train long memory, plot result
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save("model_test2.pth")
            print(f"Game: {agent.number_of_games}, Score: {score}, Record: {record}")

            plot_scores.append(score)
            total_scores += score
            mean_score = total_scores / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
