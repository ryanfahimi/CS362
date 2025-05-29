from collections import Counter

from mdp import MDP
from approach import approach

mdp = MDP(map_file="lander.actions")
start_state = "1,1"

# Value iteration demo
mdp.value_iteration()
print("Value Iteration Utilities:", mdp.utilities)
print("Value Iteration Policy:", mdp.compute_policy())
simulation = mdp.simulate_policy(start_state, 1000)
proportions = {
    key: count / len(simulation) for key, count in Counter(simulation).items()
}
print(f"Simulating Value Iteration Policy: {proportions}")

# Policy iteration demo
mdp = MDP(map_file="lander.actions")
mdp.policy_iteration()
print("Policy Iteration Utilities:", mdp.utilities)
print("Policy Iteration Policy:", mdp.compute_policy())
simulation = mdp.simulate_policy(start_state, 1000)
proportions = {
    key: count / len(simulation) for key, count in Counter(simulation).items()
}
print(f"Simulating Policy Iteration Policy: {proportions}")


# Q-learning demo for Approach (n=10)
print("\nQ-learning (Approach) Policy for n=10:")
approach(10)
