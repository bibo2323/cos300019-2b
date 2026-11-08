"""
Owner: Siti Nur Fatimah

problem.py - Route Finding problem: file parser, graph, and search-tree node.

Everyone's algorithms use these two classes:
    Problem     - the map (nodes, directed edges, origin, destinations)
    SearchNode  - one node in the SEARCH TREE (not the same as a map node)

Input file format:
    Nodes:
    1: (4,1)
    Edges:
    (2,1): 4
    Origin:
    2
    Destinations:
    5; 4
"""
import math
import re


class SearchNode:
    """A node in the search tree. 'state' is the map node number."""

    __slots__ = ("state", "parent", "g", "depth")

    def __init__(self, state, parent=None, g=0.0):
        self.state = state                      # map node number, e.g. 3
        self.parent = parent                    # SearchNode or None (root)
        self.g = g                              # path cost from origin to here
        self.depth = 0 if parent is None else parent.depth + 1

    def path(self):
        """List of map node numbers from origin to this node, e.g. [2, 3, 5]."""
        result, node = [], self
        while node is not None:
            result.append(node.state)
            node = node.parent
        result.reverse()
        return result

    def on_path(self, state):
        """True if 'state' is already on the path from the root to this node.
        Use it to avoid going round in circles (cycles)."""
        node = self
        while node is not None:
            if node.state == state:
                return True
            node = node.parent
        return False


class Problem:
    """The Route Finding problem loaded from a text file."""

    def __init__(self, nodes, edges, origin, destinations):
        self.nodes = nodes                      # {node_id: (x, y)}
        self.edges = edges                      # {(from, to): cost}  (directed)
        self.origin = origin                    # node_id
        self.destinations = destinations        # list of node_ids
        # adjacency list, children sorted ascending by node number (tie-break rule)
        self._adj = {n: [] for n in nodes}
        for (a, b), cost in edges.items():
            self._adj.setdefault(a, []).append((b, cost))
        for n in self._adj:
            self._adj[n].sort(key=lambda pair: pair[0])
        # longest edge by straight-line LENGTH between its two dots (not its cost)
        self._max_edge_len = max(
            (math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
             for a, b in edges if a in nodes and b in nodes), default=0)

    # ---- helpers for the algorithms -------------------------------------
    def root(self):
        """The starting SearchNode."""
        return SearchNode(self.origin)

    def is_goal(self, state):
        return state in self.destinations

    def expand(self, node):
        """Children of a SearchNode as a list of SearchNodes, in ASCENDING
        node-number order. Remember: nodes_created += len(children)."""
        return [SearchNode(nxt, node, node.g + cost)
                for nxt, cost in self._adj.get(node.state, [])]

    def h(self, state):
        """Heuristic: straight-line (Euclidean) distance from 'state' to the
        NEAREST destination, using the node coordinates."""
        x, y = self.nodes[state]
        return min(math.hypot(x - self.nodes[d][0], y - self.nodes[d][1])
                   for d in self.destinations)

    def h_moves(self, state):
        """Heuristic for the FEWEST MOVES (every move counts as 1), for CUS2.
        One move can cover at most 'longest edge' of distance, so
        ceil(h / longest_edge) never overestimates the number of moves."""
        if self._max_edge_len == 0:
            return 0
        return math.ceil(self.h(state) / self._max_edge_len - 1e-9)

    def path_cost(self, path):
        """Total edge cost of a path given as a list of node numbers."""
        return sum(self.edges[(a, b)] for a, b in zip(path, path[1:]))


def _number(text):
    """'4' -> 4, '4.5' -> 4.5"""
    value = float(text)
    return int(value) if value.is_integer() else value


def load_problem(filename):
    """Read a problem file and return a Problem. Raises ValueError if the file
    is badly formatted."""
    nodes, edges, origin, destinations = {}, {}, None, []
    section = None

    with open(filename, "r") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            keyword = line.lower().rstrip(":").strip()
            if keyword in ("nodes", "edges", "origin", "destinations"):
                section = keyword
                continue

            try:
                if section == "nodes":
                    m = re.match(r"(\d+)\s*:\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", line)
                    nodes[int(m.group(1))] = (_number(m.group(2)), _number(m.group(3)))
                elif section == "edges":
                    m = re.match(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*([-\d.]+)", line)
                    edges[(int(m.group(1)), int(m.group(2)))] = _number(m.group(3))
                elif section == "origin":
                    origin = int(line)
                elif section == "destinations":
                    destinations = [int(x) for x in re.split(r"[;,\s]+", line) if x]
                else:
                    raise ValueError("text before any section keyword")
            except (AttributeError, ValueError):
                raise ValueError("%s line %d: cannot read '%s'" % (filename, line_no, line))

    if origin is None or not destinations or not nodes:
        raise ValueError("%s: missing Nodes, Origin or Destinations" % filename)
    return Problem(nodes, edges, origin, destinations)