"""
Multi-goal Route Finding (Research): find the lowest-cost route that visits ALL destinations.

Imports Problem from problem.py and reads the same map files.

"""
import heapq   
import math   
import sys     

from problem import load_problem   # reuses the map reader from problem.py


class StateNode:   #one node in the search tree
    """A node in the search tree. 'visited' is a frozenset of destinations."""
    __slots__ = ("node", "visited", "parent", "g")   

    def __init__(self, node, visited, parent, g):   #runs every time a new StateNode is made
        self.node = node        
        self.visited = visited  # the destinations already visited on the way here
        self.parent = parent   
        self.g = g              #total road cost paid so far to get here

    def path(self):   
        result, cur = [], self   # start with an empty list, and begin at this node
        while cur is not None:  
            result.append(cur.node)   #add the current map node to the list
            cur = cur.parent          #step back to the node it came from
        result.reverse()   #the list was built backwards, so flip it to origin -> end
        return result     

    def state_on_path(self, node, visited):   # checks for loops
        """True if the same (node, visited set) is already on the path."""
        cur = self   # start checking from this node
        while cur is not None:  
            if cur.node == node and cur.visited == visited:  
                return True   
            cur = cur.parent   # move one step back up the path
        return False   


def _distance(problem, a, b):   # straight-line distance between two map nodes
    ax, ay = problem.nodes[a]   # x and y position of node a
    bx, by = problem.nodes[b]   # x and y position of node b
    return math.hypot(ax - bx, ay - by)   # the distance formula 


def h_nearest(problem, node, visited):   # distance to the closest destination left
    left = [d for d in problem.destinations if d not in visited]   # destinations not visited yet
    return min((_distance(problem, node, d) for d in left), default=0)   # smallest distance, or 0 if none left


def h_farthest(problem, node, visited):   # distance to the furthest destination left
    left = [d for d in problem.destinations if d not in visited]   # destinations not visited yet
    return max((_distance(problem, node, d) for d in left), default=0)   # biggest distance, or 0 if none left


def search(problem, method):   # runs one search method on the map
    """method is 'UCS', 'A*_nearest' or 'A*_farthest'.
    Returns (cost, nodes_created, path); cost is None when no route exists."""
    all_dests = frozenset(problem.destinations)   # every destination, stored as a set

    def heuristic(node, visited):   # picks which estimate to use, based on the method name
        if method == "A*_nearest":  
            return h_nearest(problem, node, visited)
        if method == "A*_farthest":  
            return h_farthest(problem, node, visited)
        return 0                                   

    start = problem.origin   # the node where the agent begins
    start_visited = frozenset([start]) & all_dests   # if the origin is also a destination, it counts as visited
    root = StateNode(start, start_visited, None, 0)   # first node: no parent, cost 0
    created = 1   #the root counts as the first created node
    counter = 0   #counts nodes added to the queue, so ties go to the one added first
    frontier = [(heuristic(start, start_visited), start, counter, root)]   # queue holding the root, with (priority, node number, counter, node)

    while frontier:   # keep searching while there are nodes waiting
        _, _, _, cur = heapq.heappop(frontier)   #take out node with lowest priority
        if cur.visited == all_dests:               # goal test when expanded
            return cur.g, created, cur.path()   #return cost, node count and route

        children = []   #will hold every node reachable from here
        for nxt, cost in problem._adj.get(cur.node, []):  
            visited = cur.visited | ({nxt} & all_dests)   
            children.append(StateNode(nxt, frozenset(visited), cur, cur.g + cost))   #make the child, with cost so far plus this road
        created += len(children)                  

        for child in children:   # decide which children go into the queue
            if cur.state_on_path(child.node, child.visited):   # skip it if the same state is already on this path
                continue   # moves on to the next child 
            counter += 1   
            priority = child.g + heuristic(child.node, child.visited)   # cost so far plus the guess of cost left
            heapq.heappush(frontier, (priority, child.node, counter, child))   #add to the queue, the smaller node number wins ties, then the earlier one

    return None, created, []   


METHODS = ["UCS", "A*_nearest", "A*_farthest"]   


def main(argv):  
    if len(argv) != 2:  
        print("Usage: python multi_goal_search.py <filename>")   
        return 1   # 1 means something went wrong
    problem = load_problem(argv[1])  
    print("%s  (origin %d, destinations %s)"  
          % (argv[1], problem.origin, "; ".join(map(str, problem.destinations))))   # file name, origin, and the destinations
    for m in METHODS:   #go through UCS, A*_nearest and A*_farthest one by one
        cost, created, path = search(problem, m)   # run the method and keep its results
        if cost is None:   #None means no route could visit every destination
            print("%-12s no route visits all destinations (nodes %d)" % (m, created))   
        else:  
            print("%-12s cost %-5s nodes %-4d path %s"   
                  % (m, cost, created, " ".join(map(str, path))))   #method name, cost, nodes created, and the route
    return 0   

if __name__ == "__main__":   
    sys.exit(main(sys.argv))  