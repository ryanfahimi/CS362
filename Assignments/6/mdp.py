import random
from collections import defaultdict, Counter


## transitions
## The transition probabilities are stored in a dictionary mapping
## (state, action) pairs to a list of edges -
## (tuples indicating destinations and probabilities)
## e.g. {("1,1","up") : [("1,1", 0.8), ("1,2", 0.2)],
##       ("1,1","down") : [("1,2", 0.8), ("1,1", 0.2)], ...


class MDP:
    def __init__(self, gamma=0.8, error=0.01, map_file=None, reward=-0.04):
        self.gamma = gamma
        self.error = error
        self.reward = reward
        if map_file:
            self.goals, self.transition_probs = load_map_from_file(map_file)
        else:
            self.goals = []
            self.transition_probs = defaultdict(list)
        self.states = set(
            [item[0] for item in self.transition_probs.keys()]
            + [item[0] for item in self.goals]
        )
        self.actions = set([item[1] for item in self.transition_probs.keys()])
        self.utilities = self.initialize_utilities()

    def __repr__(self):
        return f"Gamma: {self.gamma}\nError: {self.error}\nReward: {self.reward}\nGoals: {self.goals}\nTransitions: {self.transition_probs}\nStates: {self.states}\nActions: {self.actions}"

    def initialize_utilities(self):
        utilities = defaultdict(float)
        for state in self.states:
            utilities[state] = random.uniform(0, 1)
        for item in self.goals:
            utilities[item[0]] = float(item[1])
        return utilities

    ## return the policy, represented as a dictionary mapping states to actions, for the current utilities.
    ## you do this.
    def compute_policy(self):
        ## map states to actions.
        ## keys are self.states
        # for each state:
        # check each action - what's the EU of taking that action in that state?
        # store highest EU action in dictionary
        return {state: self.compute_eu(state)[0] for state in self.states}

    ## for a state, compute its expected utility
    def compute_eu(self, state):
        best_action = None
        ## are we at a goal?
        for goal in self.goals:
            if state == goal[0]:
                return None, goal[1]
        ## if not, for each possible action, get all the destinations and compute their EU. keep the max.
        best_eu = float("-inf")
        for action in self.actions:
            eu = 0.0
            destinations = self.transition_probs[(state, action)]
            for d in destinations:
                eu += self.utilities[d[1]] * float(d[0])
            if eu >= best_eu:
                best_action = action
                best_eu = eu
        return best_action, best_eu

    ## you do this one.
    ## 1. Initialize the utilities to random values.
    ## 2 do:
    ##     for state in states:
    ##           compute its new EU
    ##           save those in a separate array
    ##     update all values
    ##  while any EU changes by more than self.error * ( 1 - self.gamma) / self.gamma
    ## for state in states:
    ##   computePolicy.
    def value_iteration(self):
        self.utilities = self.initialize_utilities()

        converged = False
        while not converged:
            new_utilities = defaultdict(float)
            for state in self.states:
                new_utilities[state] = self.reward + self.gamma * float(
                    self.compute_eu(state)[1]
                )
            for goal in self.goals:
                new_utilities[goal[0]] = float(goal[1])

            converged = True
            for state in self.states:
                if abs(new_utilities[state] - self.utilities[state]) > (
                    self.error * (1 - self.gamma) / self.gamma
                ):
                    converged = False
                self.utilities[state] = new_utilities[state]

    ## you do this one.
    ## 1. Initialize the utilities to random values.
    ## 2. Generate a policy.
    ## do :
    ##    given the policy, update the utilities.
    ##    call computePolicy to get the policy for these utilities.
    ## while: any part of the policy changes.
    def policy_iteration(self):
        self.utilities = self.initialize_utilities()
        policy = self.compute_policy()

        while True:
            new_utilities = defaultdict(float)
            for state in self.states:
                new_utilities[state] = self.reward + self.gamma * sum(
                    self.utilities[d[1]] * float(d[0])
                    for d in self.transition_probs[(state, policy[state])]
                )
            for goal in self.goals:
                new_utilities[goal[0]] = float(goal[1])

            self.utilities = new_utilities
            new_policy = self.compute_policy()
            if new_policy == policy:
                break
            policy = new_policy

    ## You do this one.
    # use Monte Carlo simulation to test the policy.
    ## start_state is the initial state.
    ## use the generated policy along with self.transitions to
    ## generate a sequence of states until you reach one of the goals.
    ## return the list of goal states this reached - this will let you
    ## determine how frequently you are able to reach the desired goal.
    def simulate_policy(self, start_state, n_iterations):
        goal_states = []
        policy = self.compute_policy()
        for _ in range(n_iterations):
            state = start_state
            while True:
                action = policy[state]
                destinations = self.transition_probs[(state, action)]
                next_state = random.choices(
                    destinations,
                    weights=[float(d[0]) for d in destinations],
                )[0][1]
                if next_state in dict(self.goals):
                    goal_states.append(next_state)
                    break
                state = next_state
        return goal_states


def load_map_from_file(f_name):
    goals = []
    transitions = defaultdict(list)
    with open(f_name) as f:
        for line in f:
            if line.startswith("#") or len(line) < 2:
                continue
            elif line.startswith("goals"):
                goals = [tuple(x.split(":")) for x in line.split()[1:]]
            else:
                source, action, destinations = line.split(" ", 2)
                transitions[(source, action)] = [
                    tuple(x.split(":")) for x in destinations.split()
                ]
    return goals, transitions


def main():
    mdp = MDP(map_file="lander.actions")
    start_state = "1,1"
    print(mdp)

    mdp.value_iteration()
    print("\nValue Iteration:")
    print(f"Utilities: {mdp.utilities}")
    print(f"Policy: {mdp.compute_policy()}")

    simulation = mdp.simulate_policy(start_state, 1000)
    proportions = {
        key: count / len(simulation) for key, count in Counter(simulation).items()
    }
    print(f"Simulating policy: {proportions}")

    mdp.policy_iteration()
    print("\nPolicy Iteration:")
    print(f"Utilities: {mdp.utilities}")
    print(f"Policy: {mdp.compute_policy()}")

    simulation = mdp.simulate_policy(start_state, 1000)
    proportions = {
        key: count / len(simulation) for key, count in Counter(simulation).items()
    }
    print(f"Simulating policy: {proportions}")


if __name__ == "__main__":
    main()
