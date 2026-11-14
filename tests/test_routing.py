"""
tests/test_routing.py - COS30019 Assignment 2B (TBRGS).  Owner: Emily Sheryllina Aren Standly
"""
import heapq
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                    

import road_map as rm
import tbrgs_routing as tr
import travel_time as tt
from problem import Problem

ORIGIN, DEST = 2000, 3002                          
WHEN = "2006-10-27 08:00"                          


def flows_all(value):
    """Same flow (vehicles/hour) at every site."""
    sites, _ = tr.load_network()
    return {int(s): float(value) for s in sites.index}


def cheapest_time(problem, origin, goal):
    """Plain Dijkstra on the problem's costs - an independent check for A*."""
    best, heap = {origin: 0.0}, [(0.0, origin)]
    while heap:
        cost, here = heapq.heappop(heap)
        if here == goal:
            return cost
        if cost > best.get(here, float("inf")):
            continue
        for (a, b), c in problem.edges.items():
            if a == here and cost + c < best.get(b, float("inf")):
                best[b] = cost + c
                heapq.heappush(heap, (cost + c, b))
    return None


class TestRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm.save_edges()  # make sure map file current one
        cls.adj = tr.load_network()[1]

    def test_T23_routes_start_and_end_at_the_right_sites(self):
        result = tr.find_routes(ORIGIN, DEST, WHEN, model="historic")
        self.assertGreaterEqual(len(result["routes"]), 1)
        for r in result["routes"]:
            self.assertEqual(r["sites"][0], ORIGIN)
            self.assertEqual(r["sites"][-1], DEST)
            for a, b in zip(r["sites"], r["sites"][1:]):
                self.assertIn(b, self.adj[a]) # every step uses road link

    def test_T24_routes_are_cheapest_first_distinct_and_loop_free(self):
        routes = tr.find_routes(ORIGIN, DEST, WHEN, model="historic")["routes"]
        times = [r["seconds"] for r in routes]
        self.assertEqual(times, sorted(times))
        paths = [tuple(r["sites"]) for r in routes]
        self.assertEqual(len(paths), len(set(paths)))
        for p in paths:
            self.assertEqual(len(p), len(set(p)))  # no site twice in one route

    def test_T25_never_more_than_k_routes(self):
        self.assertEqual(len(tr.find_routes(ORIGIN, DEST, WHEN, k=5, model="historic")["routes"]), 5)
        self.assertEqual(len(tr.find_routes(ORIGIN, DEST, WHEN, k=2, model="historic")["routes"]), 2)
        self.assertEqual(len(tr.find_routes(ORIGIN, DEST, WHEN, k=1, model="historic")["routes"]), 1)

    def test_T26_unconnected_sites_give_no_route(self):
        problem = Problem({1: (0, 0), 2: (60, 0), 3: (200, 0)}, {(1, 2): 70, (2, 1): 70}, 1, [3])
        self.assertEqual(tr.top_k_routes(problem, 5)[0], [])
        adj = {1: {2: 1.0}, 2: {1: 1.0}, 3: {}}
        self.assertNotIn(3, tr.reachable_from(adj, 1))

    def test_T27_same_origin_and_destination(self):
        result = tr.find_routes(ORIGIN, ORIGIN, WHEN, model="historic")
        self.assertEqual(len(result["routes"]), 1)
        self.assertEqual(result["routes"][0]["sites"], [ORIGIN])
        self.assertEqual(result["routes"][0]["seconds"], 0.0)

    def test_T28_more_traffic_means_longer_travel_time(self):
        quiet = tr.find_routes(ORIGIN, DEST, WHEN, flows=flows_all(100))["routes"][0]["seconds"]
        busy = tr.find_routes(ORIGIN, DEST, WHEN, flows=flows_all(1400))["routes"][0]["seconds"]
        self.assertGreater(busy, quiet)

    def test_T29_flow_above_capacity_is_the_same_as_capacity(self):
        at_cap = tr.find_routes(ORIGIN, DEST, WHEN, flows=flows_all(tt.CAPACITY_FLOW))["routes"][0]["seconds"]
        above = tr.find_routes(ORIGIN, DEST, WHEN, flows=flows_all(4000))["routes"][0]["seconds"]
        self.assertAlmostEqual(above, at_cap, places=6)
        self.assertAlmostEqual(tt.CAPACITY_FLOW, 1500.0, delta=0.1)

    def test_T30_best_route_is_really_the_cheapest(self):
        flows = flows_all(600)
        problem = tr.build_problem(ORIGIN, DEST, flows)
        first = tr.find_routes(ORIGIN, DEST, WHEN, flows=flows)["routes"][0]["seconds"]
        self.assertAlmostEqual(first, cheapest_time(problem, ORIGIN, DEST), places=6)
        #no link is faster than straight line at 60 km/h
        for (a, b), cost in problem.edges.items():
            (xa, ya), (xb, yb) = problem.nodes[a], problem.nodes[b]
            self.assertLessEqual(((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5, cost)

    def test_T31_now_is_mapped_into_october_2006(self):
        ts, mapped = tr.to_dataset_time("2026-11-12 07:50")      # a Thursday in 2026
        self.assertTrue(mapped)
        self.assertEqual(ts, pd.Timestamp("2006-10-26 07:45"))   # Thursday 26 Oct 2006, 15-min 
        ts2, mapped2 = tr.to_dataset_time("2006-10-12 09:07")    # already inside the data
        self.assertFalse(mapped2)
        self.assertEqual(ts2, pd.Timestamp("2006-10-12 09:00"))

    def test_T33_all_six_part_a_methods_return_a_valid_route(self):
        for name in ["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"]:
            routes = tr.find_routes(ORIGIN, DEST, WHEN, flows=flows_all(600), algorithm=name)["routes"]
            self.assertGreaterEqual(len(routes), 1, name)
            sites = routes[0]["sites"]
            self.assertEqual((sites[0], sites[-1]), (ORIGIN, DEST), name)
            for a, b in zip(sites, sites[1:]):
                self.assertIn(b, self.adj[a], name)        #real road links

    def test_T34_astar_and_ucs_are_cheapest_and_no_method_beats_them(self):
        times = {name: tr.find_routes(ORIGIN, DEST, WHEN, flows=flows_all(600),
                                      algorithm=name)["routes"][0]["seconds"]
                 for name in ["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"]}
        self.assertAlmostEqual(times["AS"], times["CUS1"], places=6)
        for name, seconds in times.items():
            self.assertGreaterEqual(seconds, times["AS"] - 1e-6, name)

    def test_T35_unknown_algorithm_is_refused(self):
        with self.assertRaises(ValueError):
            tr.find_routes(ORIGIN, DEST, WHEN, flows=flows_all(600), algorithm="XYZ")


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "models", "gru.keras")),
                     "train the GRU first: python gru_model.py")
class TestGruRouting(unittest.TestCase):

    def test_T32_gru_routes_work_end_to_end(self):
        result = tr.find_routes(ORIGIN, DEST, WHEN, model="gru")
        self.assertGreaterEqual(len(result["routes"]), 1)
        flows, _ = tr.predict_flows("gru", pd.Timestamp(WHEN))
        self.assertEqual(len(flows), 40)
        self.assertTrue(all(f >= 0 for f in flows.values()))
        free = tr.find_routes(ORIGIN, DEST, WHEN, flows=flows_all(0))["routes"][0]["seconds"]
        self.assertGreaterEqual(result["routes"][0]["seconds"], free - 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
