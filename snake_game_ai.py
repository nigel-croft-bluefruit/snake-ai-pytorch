from agent import Agent
from game import SnakeGameAI
from helper import plot
import argparse
import sys


def train(model_name, reload=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(reload)
    game = SnakeGameAI()
    number_of_steps = 0

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state

        reward, done, score = game.play_step(final_move)

        state_new = agent.get_state(game)

        # train short memory
        #current_time = time.time()
        #agent.train_short_memory(state_old, final_move, reward, state_new, done)
        #delta_time = time.time() - current_time
        #print(f"Train Short Mem: {delta_time}")

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        number_of_steps = number_of_steps + 1

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1

            if agent.epsilon > 0:
                agent.epsilon -= 1

            agent.train_long_memory(number_of_steps)

            if score > record:
                record = score
                agent.model.save(model_name)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            number_of_steps = 0


def demo(filename):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(filename)
    game = SnakeGameAI()
    number_of_steps = 0

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="model.h5",
                        help='Set name of model for autosave.')
    parser.add_argument('--reload', type=str,
                        help='Load a previously saved model.')
    parser.add_argument('--demo', type=str,
                        help='Load up a demo using a saved model.')
    args = parser.parse_args(sys.argv[1:])

    if args.demo:
        demo(args.demo)
    else:
        train(args.model_name, args.reload)


if __name__ == '__main__':
    main()
