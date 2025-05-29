from math import sqrt
from queue import PriorityQueue
from Graph import Graph, Node, Edge


class MapState:
    ## f = total estimated cost
    ## g = cost so far
    ## h = estimated cost to goal
    ## location and "x,y" string
    ## mars_graph. A Graph object that tells us which squares on
    ## the map are adjacent.
    ## prev_state: a pointer back to the previous state.
    ## When we reach the goal, we'll follow that back to the start
    ## to generate the sequence of moves.
    def __init__(self, location="", mars_graph=None, prev_state=None, g=0, h=0):
        self.location = location
        self.mars_graph = mars_graph
        self.prev_state = prev_state
        self.g = g
        self.h = h
        self.f = self.g + self.h

    def __eq__(self, other):
        return self.location == other.location

    def __hash__(self):
        return hash(self.location)

    def __repr__(self):
        return "(%s)" % self.location

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def is_goal(self):
        return self.location == "1,1"

    ## You implement this.
    ## For the current state:
    ## Use the mars_graph to find all neighbors.
    # Each successor state's g is this state's g + 1.
    #
    def successors(self):
        return [
            MapState(neighbor.dest.value, self.mars_graph, self, self.g + 1, 0)
            for neighbor in self.mars_graph.g[Node(self.location)]
        ]


## You implement A* here. Use BFS and DFS as a starting point.
def a_star(start_state, heuristic_fn, goal_test, use_closed_list=True):
    search_queue = PriorityQueue()
    closed_list = {}
    state_count = 0
    search_queue.put(start_state)
    state_count += 1
    ## you do the rest.
    if use_closed_list:
        closed_list[start_state] = True
    while not search_queue.empty():
        current_state = search_queue.get()
        if goal_test(current_state):
            print("Goal found")
            print(current_state)
            print(f"Total states generated: {state_count}")
            ptr = current_state
            while ptr.prev_state is not None:
                ptr = ptr.prev_state
                print(ptr)
            return current_state
        else:
            successors = current_state.successors()
            state_count += len(successors)
            if use_closed_list:
                successors = [item for item in successors if item not in closed_list]
                for s in successors:
                    closed_list[s] = True
            for s in successors:
                s.h = heuristic_fn(s)
                s.f = s.g + s.h
                search_queue.put(s)

    print(f"Total states generated: {state_count}")


## default heuristic - we can use this to implement uniform cost search
def h1(state):
    return 0


## you do this - return the straight-line distance between the state and (1,1)
def sld(state):
    x1, y1 = map(int, state.location.split(","))
    x2, y2 = 1, 1
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


## you implement this. Open the file filename, read in each line,
## construct a Graph object and assign it to self.mars_graph().
def read_mars_graph(filename):
    graph = Graph()
    with open(filename) as f:
        for line in f:
            node_str, neighbors_str = line.split(":")
            node = Node(node_str)
            graph.add_node(node)
            edges = [
                Edge(node, Node(neighbor), 1)
                for neighbor in neighbors_str.strip().split()
            ]
            for e in edges:
                graph.add_edge(e)

    return graph


if __name__ == "__main__":
    mars_graph = read_mars_graph("MarsMap")
    start = MapState("4,4", mars_graph)
    print("A*")
    a_star(start, sld, MapState.is_goal)
    print()
    print("Uniform Cost Search")
    a_star(start, h1, MapState.is_goal)
