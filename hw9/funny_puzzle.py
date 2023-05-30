import copy
import heapq

def manhattan_dist(state):
    """
    Gets the manhattan distance of an input puzzle state.
    """

    h = sum(abs((val - 1) % 3 - i %3) + abs((val - 1) // 3 - i // 3)
        for i, val in enumerate(state) if val)

    return h 

def successors(state):
    """
    This is just a copy of print_succ. However the return is not printed. And
    is instead returned to the solve function().
    """

    # Create an empty list of successor states:
    successors = []

    # Get the index of the blank space in the grid:
    zero = state.index(0)

    # UP: if zero not on top row, swap it with tile above it
    if zero > 2:
        newState = copy.deepcopy(state)
        newState[zero] = newState[zero - 3]
        newState[zero - 3] = 0
        successors.append(newState)

    # DOWN: If zero not on bottom row, swap it with tile below it
    if zero < 6:
        newState = copy.deepcopy(state)
        newState[zero] = newState[zero + 3]
        newState[zero + 3] = 0
        successors.append(newState)

    # LEFT: If zero not in left column, swap it with tile to the left
    if zero % 3 > 0:
        newState = copy.deepcopy(state)
        newState[zero] = newState[zero - 1]
        newState[zero - 1] = 0
        successors.append(newState)

    # RIGHT: If zero not on right column, swap it with tile to the right
    if zero % 3 < 2:
        newState = copy.deepcopy(state)
        newState[zero] = newState[zero + 1]
        newState[zero + 1] = 0
        successors.append(newState)

    # Sort the list of successors, then return:
    successors.sort()
    return successors

def print_succ(state):
    """
    Print the successor states to a given input state:
    """

    # Create an empty list of successor states:
    successors = []

    # Get the index of the blank space in the grid:
    zero = state.index(0)

    # UP: if zero not on top row, swap it with tile above it
    if zero > 2:
        newState = copy.deepcopy(state)
        newState[zero] = newState[zero - 3]
        newState[zero - 3] = 0
        successors.append(newState)

    # DOWN: If zero not on bottom row, swap it with tile below it
    if zero < 6:
        newState = copy.deepcopy(state)
        newState[zero] = newState[zero + 3]
        newState[zero + 3] = 0
        successors.append(newState)

    # LEFT: If zero not in left column, swap it with tile to the left
    if zero % 3 > 0:
        newState = copy.deepcopy(state)
        newState[zero] = newState[zero - 1]
        newState[zero - 1] = 0
        successors.append(newState)

    # RIGHT: If zero not on right column, swap it with tile to the right
    if zero % 3 < 2:
        newState = copy.deepcopy(state)
        newState[zero] = newState[zero + 1]
        newState[zero + 1] = 0
        successors.append(newState)

    # Sort the list of successors, then return:
    successors.sort()
    #return successors
    for move in successors:
        print(move, "h=", manhattan_dist(move))

def solve(state):
    """
    Function to solve the 8-tile puzzle problem. Uses a priority queue, and 
    preforms an A* search on the possible puzzle states to find a winning
    solution.
    """

    # Setup open and closed queues, along with parameters:
    open_queue = []
    closed_queue = {}
    closedData = {}
    h = manhattan_dist(state)

    # Add first node to the open_queue:
    heapq.heappush(open_queue, (h, state, (0, h, -1)))

    heapE = True
    parent_index = 0
    g = 0

    # While the heap is not empty:
    while heapE:
        # Pop the previous node, and push it to the closed queue:
        curr_state = heapq.heappop(open_queue)
        
        parent_index += 1
        closed_queue[parent_index] = curr_state
        closedData[parent_index] = curr_state[1]
        
        # If the current node has a dist = 0, we have a solution. Otherwise,
        # keep trying:
        if manhattan_dist(curr_state[1]) == 0:
            # Get parents of the winning node:
            parents = []
            parent_index = curr_state[2][2]
            while parent_index != -1:
                parents.append(closed_queue[parent_index])
                parent_index = closed_queue[parent_index][2][2]

            # Print some stuff:
            for node in parents[::-1]:
                print(node[1], "h=", node[2][1], "moves=", node[2][0])
            print(curr_state[1], "h=", curr_state[2][1], "moves=", curr_state[2][0])
            heapE = False

        #Else get some new successor states and keep looking:
        successors_states = successors(curr_state[1])
        g = curr_state[2][0] + 1
        for move in successors_states:
            if not (move in open_queue or move in closedData.values()):
                h = manhattan_dist(move)
                heapq.heappush(open_queue, (g + h, move, (g, h, parent_index)))