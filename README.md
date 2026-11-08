# COS30019 A2A - Route Finding (Tree-Based Search)

## Run
    python search.py <filename> <method>
    python search.py tests\PathFinder-test-1.txt BFS
Methods: DFS, BFS, GBFS, AS, CUS1, CUS2  (or use search.bat on Windows)

## Batch test (all methods x all files, saves results.csv)
    python run_all.py tests results.csv

## Files
- search.py          command-line entry + output format
- problem.py         parser, Problem (graph) and SearchNode classes
- algorithms/        dfs_bfs.py, gbfs_astar.py,S custom.py (one file per person)
- run_all.py         batch tester for the Testing / Insights sections
- tests/             test case files (.txt)

## Rules for every algorithm
1. One function per method: def name(problem)
2. Return (goal, nodes_created, path); if no goal: (None, nodes_created, [])
3. Use problem.root(), problem.expand(node), problem.is_goal(state), problem.h(state)
4. Count the root as a created node; add len(children) each time you expand
5. Tie-break: smaller node number first, then the one added first
6. Stop when the first goal node is expanded

# Research Extension - Visit ALL Destinations

## Run
    python multi_goal_search.py <filename>
    python multi_goal_search.py allgoal_tests\ag4_four.txt
Runs UCS, A* (nearest heuristic) and A* (farthest heuristic) on the map.
It does not change search.py or any submitted method.

## Files
- multi_goal_search.py  research extension (visit all destinations)
- allgoal_tests/        test maps created for the research extension

## Rules for the research extension
1. Put multi_goal_search.py in the same folder as problem.py
2. Keep the research maps in allgoal_tests\, not in tests\ (so run_all.py does not include them)
3. A state is the current node and the set of destinations visited so far
4. The goal is reached when every destination has been visited
5. Count the root as a created node; add len(children) each time you expand
6. Tie-break: smaller node number first, then the one added first