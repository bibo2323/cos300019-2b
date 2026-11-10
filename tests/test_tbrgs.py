"""
tests/test_tbrgs.py - COS30019 Assignment 2B (TBRGS)

Automatic test cases for: data processing (T01-T06), travel time (T07-T13)
and the road map (T14-T18). Teammates add their own tests below (ML models,
search integration, GUI) using the same style, e.g. T19, T20 ...

Run from the repo root and keep a record for the report:
    python tests/test_tbrgs.py > tests/test_results.txt 2>&1
"""
import os
import sys
import unittest

import pandas as pd

# let the tests import the files that sit in the repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                     # data/ paths are relative to the root

import data_processing as dp
import road_map as rm
import travel_time as tt


class TestDataProcessing(unittest.TestCase):
    """T01-T06: the SCATS data is read, cleaned and reshaped correctly."""

    @classmethod
    def setUpClass(cls):
        cls.long15, cls.hourly, cls.sites, cls.report = dp.process()

    def test_T01_forty_sites_thirty_one_days(self):
        self.assertEqual(self.report["sites"], 40)
        self.assertEqual(self.report["days"], 31)

    def test_T02_duplicates_removed(self):
        self.assertEqual(self.report["duplicate_rows_removed"], 31)
        self.assertEqual(self.report["clean_rows"], 4161)

    def test_T03_every_site_day_has_96_intervals(self):
        counts = self.long15.groupby(["site", "date"]).size()
        self.assertTrue((counts == dp.INTERVALS_PER_DAY).all())

    def test_T04_no_negative_or_missing_flow(self):
        self.assertFalse(self.long15["flow_15min"].isna().any())
        self.assertTrue((self.long15["flow_15min"] >= 0).all())

    def test_T05_hourly_is_sum_of_four_15min_slots(self):
        row = self.hourly.iloc[0]
        start = pd.Timestamp(row["hour_start"])
        part = self.long15[(self.long15["site"] == row["site"]) &
                           (self.long15["timestamp"] >= start) &
                           (self.long15["timestamp"] < start + pd.Timedelta(hours=1))]
        self.assertEqual(len(part), 4)
        self.assertEqual(part["flow_15min"].sum(), row["flow_per_hour"])

    def test_T06_site_coordinates_are_inside_boroondara(self):
        # site 4266 has a (0, 0) detector in the raw file; it must be ignored
        self.assertTrue(self.sites["lat"].between(-37.9, -37.7).all())
        self.assertTrue(self.sites["lon"].between(144.9, 145.2).all())


class TestTravelTime(unittest.TestCase):
    """T07-T13: flow -> speed -> travel time follows the conversion document."""

    def test_T07_low_flow_gives_speed_limit(self):
        for flow in (0, 100, 351):
            self.assertAlmostEqual(tt.flow_to_speed(flow), 60.0, delta=0.2)

    def test_T08_medium_flow_is_slower_than_free_flow(self):
        speed = tt.flow_to_speed(1000)
        self.assertGreater(speed, 32.0)
        self.assertLess(speed, 60.0)

    def test_T09_flow_above_capacity_is_treated_as_capacity(self):
        self.assertAlmostEqual(tt.flow_to_speed(5331), 32.0, delta=0.1)

    def test_T10_zero_distance_costs_only_the_intersection_delay(self):
        self.assertEqual(tt.edge_travel_time_seconds(0.0, 800), tt.INTERSECTION_DELAY_S)

    def test_T11_travel_time_never_decreases_when_flow_increases(self):
        times = [tt.edge_travel_time_seconds(1.0, f) for f in range(0, 1501, 50)]
        self.assertEqual(times, sorted(times))

    def test_T12_distance_between_sites_2000_and_3002(self):
        sites = dp.load_sites().set_index("site")
        d = tt.haversine_km(sites.loc[2000, "lat"], sites.loc[2000, "lon"],
                            sites.loc[3002, "lat"], sites.loc[3002, "lon"])
        self.assertTrue(6.5 < d < 8.0)

    def test_T13_negative_flow_is_treated_as_zero(self):
        self.assertAlmostEqual(tt.flow_to_speed(-50), 60.0, delta=0.2)


class TestRoadMap(unittest.TestCase):
    """T14-T18: the graph used by the search is complete and sensible."""

    @classmethod
    def setUpClass(cls):
        cls.edges = rm.save_edges()
        cls.adj = rm.build_adjacency(cls.edges)

    def test_T14_all_forty_sites_are_in_the_graph(self):
        self.assertEqual(len(self.adj), 40)

    def test_T15_main_network_connected_and_isolated_sites_have_no_links(self):
        # tutor: sites with no shared road stay unconnected (isolated)
        parts = sorted(rm.connected_components(self.adj), key=len, reverse=True)
        self.assertGreater(len(parts[0]), 30)               # one big network
        for small in parts[1:]:
            self.assertEqual(len(small), 1)                 # the rest are single sites
            site = next(iter(small))
            self.assertEqual(self.adj[site], {})

    def test_T16_every_link_works_in_both_directions(self):
        for a, neighbours in self.adj.items():
            for b, dist in neighbours.items():
                self.assertEqual(self.adj[b][a], dist)

    def test_T17_route_exists_from_2000_to_3002(self):
        part = next(p for p in rm.connected_components(self.adj) if 2000 in p)
        self.assertIn(3002, part)

    def test_T18_distances_are_positive_and_realistic(self):
        self.assertTrue((self.edges["distance_km"] > 0).all())
        self.assertTrue((self.edges["distance_km"] < 10).all())


# ---- teammates: add your test classes here (ML models, search, GUI) --------


if __name__ == "__main__":
    unittest.main(verbosity=2)
