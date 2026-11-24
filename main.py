import random
import numpy as np
import gymnasium as gym


env = gym.make("Taxi-v4")

alpha = 0.9  # learning rate
gamma = 0.95  # discount factor

epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01

num_episodes = 10000
max_steps = 100

# init q table (500 states x 6 actions)
q_table = np.zeros((env.observation_space.n, env.action_space.n))


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])


for episode in range(num_episodes):
    state, _ = env.reset()

    done = False

    for step in range(max_steps):
        action = choose_action(state)

        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])

        # no future value if the episode has ended
        q_table[state, action] = (1 - alpha) * old_value + alpha * (
            reward + gamma * next_max * (not done)
        )
        state = next_state

        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()


env = gym.make("Taxi-v4", render_mode="ansi")

# the 4 coloured stops on the Taxi map: name and (row, col)
locations = [("Red", (0, 0)), ("Green", (0, 4)), ("Yellow", (4, 0)), ("Blue", (4, 3))]

for episode in range(10):
    state, _ = env.reset()

    done = False
    total_reward = 0

    # decode the state number into taxi position, passenger stop, destination stop
    taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)
    pick_name, pick_pos = locations[pass_idx]
    drop_name, drop_pos = locations[dest_idx]

    print("Episode : ", episode)
    print("  Taxi starts at :", (taxi_row, taxi_col))
    print("  Pick up at     :", pick_name, pick_pos)
    print("  Drop off at    :", drop_name, drop_pos)

    for step in range(max_steps):
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state

        if done or truncated:
            print("Finished episode", episode, "with total reward", total_reward)
            break

env.close()