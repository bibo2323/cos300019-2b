"""Collects every search method so search.py can look them up by name.

Each algorithm is ONE function:   def name(problem): ...
and returns a tuple:              (goal, nodes_created, path)
    goal           - the destination node reached
    nodes_created  - how many SearchNodes were created (include the root)
    path           - list of node numbers, e.g. [2, 3, 5]
If no goal can be reached, return (None, nodes_created, []).
"""
from .dfs_bfs import dfs, bfs
from .gbfs_astar import gbfs, astar
from .custom import cus1, cus2

METHODS = {
    "DFS": dfs,
    "BFS": bfs,
    "GBFS": gbfs,
    "AS": astar,
    "CUS1": cus1,
    "CUS2": cus2,
}
