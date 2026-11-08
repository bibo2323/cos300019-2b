"""
Custom Search Algorithms
Owner: Siu Han

CUS1: Uniform Cost Search (UCS) - uninformed search,
prioritising the lowest accumulated edge cost.

CUS2: Iterative Deepening A* (IDA*) - informed search,
finding a path with the fewest moves using h_moves().

Both methods count every child created by expand(),
including children skipped by the cycle check.

IDA* also counts nodes re-created in every iteration.
"""

import heapq
import math

def cus1(problem):
    """Uniform Cost Search - Uninformed Search"""

    # 1. Get the starting node
    root = problem.root()

    # 2. Count the root node
    created = 1

    # 3. Create a priority queue
    frontier = []
    counter = 0

    heapq.heappush(
        frontier,
        (root.g, root.state, counter, root)
    )
    counter += 1

    # 4. Continue until the frontier is empty
    while frontier:

        # Get the node with the lowest path cost
        cost, _, _, node = heapq.heappop(frontier)

        # 5. Check whether the node is a goal
        if problem.is_goal(node.state):
            return node.state, created, node.path()

        # 6. Generate children
        children = problem.expand(node)

        # Count every generated child, including skipped ones
        created += len(children)

        # 7. Add valid children to the priority queue
        for child in children:

            # Avoid cycles in the current path
            if node.on_path(child.state):
                continue

            heapq.heappush(
                frontier,
                (child.g, child.state, counter, child)
            )
            counter += 1

    # No path found
    return None, created, []



def cus2(problem):
    """CUS2: Iterative Deepening A* (IDA*) - Fewest Moves"""

    created = 0

    # Recursive depth-first search with an f-value threshold
    def search(node, threshold):
        nonlocal created

        # g(n): number of moves already taken
        g = node.depth

        # h(n): lower-bound estimate of remaining moves
        h = problem.h_moves(node.state)

        # f(n) = g(n) + h(n)
        f = g + h

        # Stop exploring this branch if f exceeds threshold
        if f > threshold:
            return None, f

        # Check whether the current node is a destination
        if problem.is_goal(node.state):
            return node, math.inf
        
        # Generate all children
        all_children = problem.expand(node)

        # Count every child, including those skipped by cycle check
        created += len(all_children)

        # Keep only children that are not on the current path
        children = [
            child for child in all_children
            if not node.on_path(child.state)
        ]

        # Track the smallest f that exceeds the threshold
        minimum_exceeded = math.inf

        for child in children:

            # Continue depth-first search
            goal, next_threshold = search(child, threshold)

            if goal is not None:
                return goal, next_threshold

            if next_threshold < minimum_exceeded:
                minimum_exceeded = next_threshold

        return None, minimum_exceeded

    # Create the first root node
    root = problem.root()
    created += 1

    # Initial threshold = f(start)
    threshold = root.depth + problem.h_moves(root.state)

    # Repeat the search with increasing thresholds
    while True:

        goal, next_threshold = search(root, threshold)

        # A destination has been found
        if goal is not None:
            return goal.state, created, goal.path()

        # No higher threshold is possible: unreachable
        if next_threshold == math.inf:
            return None, created, []

        # Increase threshold to the smallest exceeded f
        threshold = next_threshold

        # Start another iteration from a new root
        root = problem.root()

        # Count root again because it is created again
        created += 1

