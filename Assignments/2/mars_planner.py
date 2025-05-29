from copy import deepcopy

from search_algorithms import (
    breadth_first_search,
    depth_first_search,
    iterative_deepening_search,
)


class RoverState:
    def __init__(
        self,
        rover_loc="station",
        sample_loc="site",
        holding_sample=False,
        charged=False,
        holding_tool=False,
        prev=None,
    ):
        self.rover_loc = rover_loc
        self.sample_loc = sample_loc
        self.holding_sample = holding_sample
        self.charged = charged
        self.holding_tool = holding_tool
        self.prev = prev
        self.depth = 0

    def __eq__(self, other):
        return (
            self.rover_loc == other.rover_loc
            and self.sample_loc == other.sample_loc
            and self.holding_sample == other.holding_sample
            and self.charged == other.charged
            and self.holding_tool == other.holding_tool
        )

    def __repr__(self):
        return (
            f"Rover Location: {self.rover_loc}\n"
            + f"Sample Location: {self.sample_loc}\n"
            + f"Holding Sample?: {self.holding_sample}\n"
            + f"Charged? {self.charged}\n"
            + f"Holding Tool? {self.holding_tool}"
        )

    def __hash__(self):
        return self.__repr__().__hash__()

    def successors(self, list_of_actions):
        ## Apply each function in the list of actions to the current state to get
        ## a new state. Add the name of the function as well.
        succ = [(item(self), item.__name__) for item in list_of_actions]

        ## remove actions that have no effect
        succ = [item for item in succ if not item[0] == self]
        return succ


def move_to_station(state):
    r2 = deepcopy(state)
    r2.rover_loc = "station"
    if r2.holding_sample:
        r2.sample_loc = "station"
    r2.prev = state
    return r2


def move_to_site(state):
    r2 = deepcopy(state)
    r2.rover_loc = "site"
    if r2.holding_sample:
        r2.sample_loc = "site"
    r2.prev = state
    return r2


def move_to_battery(state):
    r2 = deepcopy(state)
    r2.rover_loc = "battery"
    if r2.holding_sample:
        r2.sample_loc = "battery"
    r2.prev = state
    return r2


def pick_up_tool(state):
    r2 = deepcopy(state)
    r2.holding_tool = True
    r2.prev = state
    return r2


def drop_tool(state):
    r2 = deepcopy(state)
    r2.holding_tool = False
    r2.prev = state
    return r2


def use_tool(state):
    r2 = deepcopy(state)
    if r2.holding_tool and r2.rover_loc == r2.sample_loc:
        r2.holding_sample = True
    r2.prev = state
    return r2


def pick_up_sample(state):
    r2 = deepcopy(state)
    if state.rover_loc == state.sample_loc:
        r2.holding_sample = True
    r2.prev = state
    return r2


def drop_sample(state):
    r2 = deepcopy(state)
    if state.rover_loc == state.sample_loc:
        r2.holding_sample = False
    r2.prev = state
    return r2


def charge(state):
    r2 = deepcopy(state)
    if state.rover_loc == "battery":
        r2.charged = True
    r2.prev = state
    return r2


action_list = [
    pick_up_tool,
    move_to_site,
    use_tool,
    move_to_station,
    drop_tool,
    drop_sample,
    move_to_battery,
    charge,
]


def battery_goal(state):
    return state.rover_loc == "battery"


def charged_goal(state):
    return state.charged


def sample_goal(state):
    return state.sample_loc == "station"


def mission_complete(state):
    return battery_goal(state) and charged_goal(state) and sample_goal(state)


if __name__ == "__main__":
    print("Running search algorithms")
    print("Breadth First Search")
    breadth_first_search(RoverState(), action_list, mission_complete)
    print("Depth First Search")
    depth_first_search(RoverState(), action_list, mission_complete)
    print("Depth Limited Search (limit=8)")
    depth_first_search(RoverState(), action_list, mission_complete, limit=8)
    print("Iterative Deepening Search")
    iterative_deepening_search(RoverState(), action_list, mission_complete)
