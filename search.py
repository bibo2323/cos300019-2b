"""
Owner: Siti Nur Fatimah

COS30019 Assignment 2A - Route Finding (tree-based search)

Usage:   python search.py <filename> <method>
Methods: DFS  BFS  GBFS  AS  CUS1  CUS2

Output:  <filename> <method>
         <goal> <number_of_nodes>
         <path>
"""
import sys

from problem import load_problem
from algorithms import METHODS


def format_path(path):
    """How the path is printed, e.g. '2 3 5'. Change here if the lecturer
    wants another style (e.g. '2 -> 3 -> 5')."""
    return " ".join(str(n) for n in path)


def solve(problem, method):
    """Run one method. Returns (goal, nodes_created, path)."""
    result = METHODS[method](problem)
    if result is None:                           # tolerate a plain None
        return None, 0, []
    return result


def main(argv):
    if len(argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print("Methods: " + "  ".join(METHODS))
        return 1

    filename, method = argv[1], argv[2].upper()
    if method not in METHODS:
        print("Unknown method '%s'. Choose from: %s" % (argv[2], "  ".join(METHODS)))
        return 1

    try:
        problem = load_problem(filename)
    except (OSError, ValueError) as err:
        print("Error: %s" % err)
        return 1

    try:
        goal, created, path = solve(problem, method)
    except NotImplementedError as err:
        print("%s %s" % (filename, method))
        print(err)
        return 1

    print("%s %s" % (filename, method))
    if goal is None:
        print("No goal is reachable; %d" % created)
    else:
        print("%s %d" % (goal, created))
        print(format_path(path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
