from mars_planner import RoverState, action_list, mission_complete
from search_algorithms import (
    breadth_first_search,
    depth_first_search,
    iterative_deepening_search,
)
from route_finder import read_mars_graph, a_star, sld, h1, MapState


def demo_mars_planner():
    print("=== Mars Planner ===")
    init_state = RoverState()

    # Run BFS
    print("\nBreadth-First Search")
    breadth_first_search(init_state, action_list, mission_complete)

    print("\nDepth-First Search")
    depth_first_search(init_state, action_list, mission_complete)

    print("\nDepth-Limited Search (limit=8)")
    depth_first_search(init_state, action_list, mission_complete, limit=8)

    print("\nIterative Deepening Search")
    iterative_deepening_search(init_state, action_list, mission_complete)


def demo_route_finder():
    print("\n=== Route Finder ===")
    mars_graph = read_mars_graph("MarsMap")
    start = MapState("4,4", mars_graph)

    print("\nA*")
    a_star(start, sld, MapState.is_goal)

    print("\nUniform Cost Search")
    a_star(start, h1, MapState.is_goal)


if __name__ == "__main__":
    demo_mars_planner()
    demo_route_finder()
