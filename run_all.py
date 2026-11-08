"""
Owner: Siti Nur Fatimah

Batch tester - runs EVERY method on EVERY test file and saves a results table.
Use it for the Testing and Insights sections of the report.

Usage:   python run_all.py [folder] [output.csv]
Default: python run_all.py tests results.csv
"""
import csv
import glob
import os
import sys
import time

from problem import load_problem
from algorithms import METHODS
from search import solve, format_path


def main(argv):
    folder = argv[1] if len(argv) > 1 else "tests"
    out_csv = argv[2] if len(argv) > 2 else "results.csv"
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        print("No .txt test files found in '%s'" % folder)
        return 1

    rows = []
    header = ["file", "method", "goal", "nodes_created", "path_cost",
              "path_length", "time_ms", "path", "status"]
    print("%-22s %-5s %-5s %-6s %-8s %-6s %-8s %s" %
          ("file", "meth", "goal", "nodes", "cost", "moves", "ms", "path"))

    for filename in files:
        name = os.path.basename(filename)
        problem = load_problem(filename)
        for method in METHODS:
            try:
                start = time.perf_counter()
                goal, created, path = solve(problem, method)
                ms = (time.perf_counter() - start) * 1000
            except NotImplementedError:
                rows.append([name, method, "", "", "", "", "", "", "not implemented"])
                print("%-22s %-5s not implemented" % (name, method))
                continue
            except RecursionError:
                rows.append([name, method, "", "", "", "", "", "", "recursion error"])
                print("%-22s %-5s RECURSION ERROR" % (name, method))
                continue

            if goal is None:
                rows.append([name, method, "", created, "", "", "%.3f" % ms, "", "no path"])
                print("%-22s %-5s no path (nodes %d)" % (name, method, created))
            else:
                cost = problem.path_cost(path)
                rows.append([name, method, goal, created, cost, len(path) - 1,
                             "%.3f" % ms, format_path(path), "ok"])
                print("%-22s %-5s %-5s %-6d %-8s %-6d %-8.3f %s" %
                      (name, method, goal, created, cost, len(path) - 1, ms, format_path(path)))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("\nSaved %d rows to %s" % (len(rows), out_csv))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
