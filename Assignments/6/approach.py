from random import randint
from random import random

# approach.py


# This program uses reinforcement learning to determine the optimal policy
# for Approach.
# Recall that approach works like this:
# Both players agree on a limit n.
# Player 1 rolls first. They go until they either exceed n or hold.
# Then player 2 rolls. They go until they either exceed n or beat player 1's score.
# The player who is closest to n without going over wins.
# Note:
# We can reduce this to the problem of player 1 choosing the best value at which to hold.
# This is called a policy; once we know the best number to hold at, we can act optimally.


def approach(n):
    q_table = [[random() / 100.0, random() / 100.0] for i in range(n)]

    epsilon = 0.1
    alpha = 0.1
    gamma = 0.8

    for i in range(100000):
        # Select an initial state.
        state = randint(0, n - 1)
        history = []

        # play until P1 holds or busts
        while True:
            # Take the best move with p=epsilon, and the worst move with p=1-epsilon
            if random() < epsilon:
                # explore: pick the other action
                action = 1 if q_table[state][0] >= q_table[state][1] else 0
            else:
                # exploit: pick the best action
                action = 0 if q_table[state][0] >= q_table[state][1] else 1

            history.append((state, action))

            if action == 0:
                # hold: end P1’s turn
                break

            roll = randint(1, 6)
            state = state + roll

            if state > n:
                # bust
                break
            if state == n:
                # exactly hit n, treat like holding
                break

        # Continue playing until the game is done.
        # If you win, reward = 1.
        # If you lose, reward = 0.
        # simulate P2 to get reward
        if history[-1][1] == 0 and state <= n:
            # P1 held at s ≤ n, now P2 rolls
            p2 = 0
            while True:
                r2 = randint(1, 6)
                p2 += r2
                if p2 > n or p2 > state:
                    break
            reward = 1 if p2 > n else 0
        else:
            # P1 busts or never held
            reward = 0

        # Use Q-learning to update the q-table for each state-action pair visited.
        for s_prev, a_prev in reversed(history):
            # estimate of future best action
            future_best_action = 0
            if s_prev <= n:
                future_best_action = max(q_table[s_prev]) if s_prev < n else 0
            q_old = q_table[s_prev][a_prev]
            q_table[s_prev][a_prev] = q_old + alpha * (
                reward + gamma * future_best_action - q_old
            )

    ## After 100000 iterations, print out your q-table.
    for state in range(n):
        hold_q, roll_q = q_table[state]
        best = "hold" if hold_q >= roll_q else "roll"
        print(f"{state:2d}: {hold_q:.6f} {roll_q:.6f} [{best}]")


if __name__ == "__main__":
    approach(10)
