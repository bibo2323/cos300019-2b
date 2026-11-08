"""DFS and BFS (uninformed). Owner: Shirleen Teo Shu Enn

BFS and DFS (uninformed search)
Tie-breaking: problem.expand() already returns children in ascending node order.
BFS = queue (first in, first out).  DFS = stack (last in, first out), so for DFS
you must push children in REVERSE order to expand the smallest node first.
"""
from collections import deque


def bfs(problem):
    root = problem.root()
    created = 1                                  # the root counts as created
    frontier = deque([root])
    while frontier:
        node = frontier.popleft()
        if problem.is_goal(node.state):          # goal test when node is expanded
            return node.state, created, node.path()
        children = problem.expand(node)
        created += len(children)                 # count EVERY child expand() creates
        for child in children:
            if node.on_path(child.state):        # skip cycles (still counted above)
                continue
            frontier.append(child)
    return None, created, []


def dfs(problem):
    root = problem.root()
    created = 1                                  # the root counts as created
    frontier = [root]                            # a plain list used as a stack (LIFO)
    while frontier:
        node = frontier.pop()                    # take the most recently added node
        if problem.is_goal(node.state):          # goal test when node is expanded
            return node.state, created, node.path()
        # expand() gives children in ascending order; push them in REVERSE so the
        # smallest node number ends up on top of the stack and is expanded first
        children = problem.expand(node)
        created += len(children)                 # count EVERY child expand() creates
        for child in reversed(children):
            if node.on_path(child.state):        # skip cycles (still counted above)
                continue
            frontier.append(child)
    return None, created, []