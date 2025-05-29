from collections import deque

STATE_COUNT = 0


def breadth_first_search(start_state, action_list, goal_test, use_closed_list=True):
    search_queue = deque()
    closed_list = {}
    global STATE_COUNT
    STATE_COUNT = 0

    search_queue.append((start_state, ""))
    STATE_COUNT += 1
    if use_closed_list:
        closed_list[start_state] = True
    while len(search_queue) > 0:
        # this is a (state, "action") tuple
        current_state = search_queue.popleft()
        if goal_test(current_state[0]):
            print("Goal found")
            print(current_state)
            print(f"Total states generated: {STATE_COUNT}")
            ptr = current_state[0]
            while ptr.prev is not None:
                ptr = ptr.prev
                print(ptr)
                print()
            return current_state
        else:
            successors = current_state[0].successors(action_list)
            STATE_COUNT += len(successors)
            if use_closed_list:
                successors = [item for item in successors if item[0] not in closed_list]
                for s in successors:
                    closed_list[s[0]] = True
            search_queue.extend(successors)
    print(f"Total states generated: {STATE_COUNT}")


def depth_first_search(
    start_state, action_list, goal_test, use_closed_list=True, limit=0
):
    # Initialize the search stack and closed list.
    search_stack = deque()
    closed_list = {}
    global STATE_COUNT
    STATE_COUNT = 0

    start_state.depth = 0
    search_stack.append((start_state, ""))
    STATE_COUNT += 1
    if use_closed_list:
        closed_list[start_state] = True

    while search_stack:
        current_state = search_stack.pop()
        depth = current_state[0].depth
        if goal_test(current_state[0]):
            print("Goal found")
            print(current_state)
            print(f"Total states generated: {STATE_COUNT}")
            ptr = current_state[0]
            while ptr.prev is not None:
                ptr = ptr.prev
                print(ptr)
                print()
            return current_state
        else:
            if limit != 0 and depth > limit:
                continue
            successors = current_state[0].successors(action_list)
            STATE_COUNT += len(successors)

            if use_closed_list:
                successors = [item for item in successors if item[0] not in closed_list]
                for s in successors:
                    closed_list[s[0]] = True

            for s in successors:
                s[0].depth = depth + 1
                search_stack.append(s)

    print(f"Total states generated: {STATE_COUNT}")


def iterative_deepening_search(
    start_state, action_list, goal_test, use_closed_list=True, max_limit=None
):
    total_state_count = 0
    limit = 1
    while max_limit is None or limit <= max_limit:
        result = depth_first_search(
            start_state, action_list, goal_test, use_closed_list, limit=limit
        )
        total_state_count += STATE_COUNT
        if result is not None:
            print("Goal found")
            print(result)
            print(f"Total states generated: {total_state_count}")
            return result
        limit += 1

    print(f"Total states generated: {total_state_count}")
