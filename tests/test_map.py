import html as htmllib
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

# let the tests import the files that sit in the repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                     # data/ paths are relative to the root

try:
    import folium
    import map_base
    import map_routes
    HAS_FOLIUM = True
except ImportError:                                # folium not installed
    HAS_FOLIUM = False

import tbrgs_routing as tr

WHEN = "2006-10-27 08:00"                          # a Friday inside the October 2006 data
FIXED_FLOW = 600.0                                 # veh/h for every site
GREY = "#808080"


def fixed_flows():
    sites, _ = tr.load_network()
    return {int(s): FIXED_FLOW for s in sites.index}


def routes_for(origin, destination, k):
    return tr.find_routes(origin, destination, WHEN, k, flows=fixed_flows())


def read_html(path):
    """HTML of a saved map with entities decoded (so '-&gt;' becomes '->')."""
    with open(str(path), encoding="utf-8") as f:
        return htmllib.unescape(f.read())


def route_numbers(text):
    """Set of route numbers that appear as 'Route n: x min, y km' in the map."""
    return {int(n) for n in re.findall(r"Route (\d+): [\d.]+ min, [\d.]+ km", text)}


def route_colours(text):
    """Route colours (not grey roads / blue sites) used by lines in the map."""
    used = set(re.findall(r'"color": "(#[0-9a-fA-F]{6})"', text))
    return used & set(map_routes.ROUTE_COLOURS)


def write_sites(path, rows):
    pd.DataFrame(rows, columns=["site", "n_approaches", "name", "lat", "lon"]).to_csv(path, index=False)


def write_edges(path, rows):
    pd.DataFrame(rows, columns=["a", "b", "distance_km", "road", "kind"]).to_csv(path, index=False)


@unittest.skipUnless(HAS_FOLIUM, "folium is not installed:  python -m pip install folium")
class TestBaseMap(unittest.TestCase):
    """M01-M08: Siti's base map."""

    @classmethod
    def setUpClass(cls):
        cls.sites = pd.read_csv(map_base.SITES_FILE)
        cls.edges = pd.read_csv(map_base.EDGES_FILE)
        cls.fmap, cls.coords = map_base.create_base_map()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "base_map.html")
        cls.fmap.save(cls.path)
        cls.text = read_html(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_M01_returns_folium_map_and_coordinate_dict(self):
        self.assertIsInstance(self.fmap, folium.Map)
        self.assertIsInstance(self.coords, dict)

    def test_M02_html_file_is_created_and_not_empty(self):
        self.assertTrue(os.path.exists(self.path))
        self.assertGreater(os.path.getsize(self.path), 10000)
        self.assertIn("leaflet", self.text.lower())

    def test_M03_every_valid_site_has_the_coordinate_from_sites_csv(self):
        for row in self.sites.itertuples(index=False):
            if map_base.is_valid_coordinate(row.lat, row.lon):
                lat, lon = self.coords[int(row.site)]
                self.assertAlmostEqual(lat, row.lat, places=6)
                self.assertAlmostEqual(lon, row.lon, places=6)

    def test_M04_all_coordinates_are_inside_boroondara(self):
        self.assertGreater(len(self.coords), 0)
        for site, (lat, lon) in self.coords.items():
            self.assertTrue(-37.95 <= lat <= -37.70, "site %d lat %s" % (site, lat))
            self.assertTrue(144.95 <= lon <= 145.20, "site %d lon %s" % (site, lon))

    def test_M05_one_marker_per_site_and_one_grey_line_per_link(self):
        info = self.fmap.base_info
        markers = [c for c in self.fmap._children.values()
                   if isinstance(c, folium.CircleMarker)]
        lines = [c for c in self.fmap._children.values()
                 if isinstance(c, folium.PolyLine)]
        self.assertEqual(len(markers), len(self.coords))
        self.assertEqual(len(markers), info["sites_drawn"])
        self.assertEqual(len(lines), info["links_drawn"])
        self.assertEqual(len(lines), len(self.edges))     # every link has two valid ends

    def test_M06_road_lines_are_grey(self):
        self.assertGreaterEqual(self.text.count('"color": "%s"' % GREY), len(self.edges))

    def test_M07_every_site_is_in_the_saved_html(self):
        for site, (lat, lon) in self.coords.items():
            self.assertIn(str(lat), self.text, "site %d is missing from the map" % site)

    def test_M08_only_one_layer_control(self):
        # map_routes adds its own layers; two layer controls would show two boxes
        controls = [c for c in self.fmap._children.values()
                    if isinstance(c, folium.LayerControl)]
        self.assertEqual(len(controls), 1)


@unittest.skipUnless(HAS_FOLIUM, "folium is not installed:  python -m pip install folium")
class TestInvalidCoordinates(unittest.TestCase):
    """M09-M13: bad coordinates must be skipped safely (use COPIES of the csv files)."""

    GOOD = [
        (1001, 4, "WARRIGAL_RD N of HIGH STREET_RD", -37.8673, 145.0915),
        (1002, 4, "WARRIGAL_RD N of TOORAK_RD", -37.8519, 145.0943),
        (1003, 2, "UNION_RD N of MAROONDAH_HWY", -37.8165, 145.0980),
    ]
    BAD = [
        (2001, 2, "blank coordinates", None, None),
        (2002, 2, "zero zero", 0.0, 0.0),
        (2003, 2, "far away (Kuching)", 1.5533, 110.3592),
        (2004, 2, "lat and lon swapped", 145.0915, -37.8673),
        (2005, 2, "text instead of numbers", "abc", "xyz"),
        (2006, 2, "lat out of range", -137.5, 145.0),
    ]
    EDGES = [
        (1001, 1002, 1.2, "WARRIGAL_RD", "road"),
        (1002, 1003, 1.5, "UNION_RD", "road"),
        (1001, 2002, 9.0, "BAD_RD", "road"),          # link to a bad site
        (2001, 2003, 9.0, "BAD_RD", "road"),           # link between two bad sites
    ]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sites_file = os.path.join(self.tmp.name, "sites.csv")
        self.edges_file = os.path.join(self.tmp.name, "edges.csv")
        write_sites(self.sites_file, self.GOOD + self.BAD)
        write_edges(self.edges_file, self.EDGES)

    def tearDown(self):
        self.tmp.cleanup()

    def test_M09_each_bad_coordinate_type_is_rejected(self):
        for _, _, name, lat, lon in self.BAD:
            self.assertFalse(map_base.is_valid_coordinate(lat, lon), name)
        self.assertFalse(map_base.is_valid_coordinate(float("nan"), -37.8))
        self.assertFalse(map_base.is_valid_coordinate(None, None))

    def test_M10_map_is_still_created_and_bad_sites_are_skipped(self):
        fmap, coords = map_base.create_base_map(sites_file=self.sites_file,
                                                edges_file=self.edges_file)
        self.assertEqual(sorted(coords), [1001, 1002, 1003])
        self.assertEqual(sorted(fmap.base_info["skipped_sites"]),
                         [2001, 2002, 2003, 2004, 2005, 2006])

    def test_M11_links_to_bad_sites_are_not_drawn(self):
        fmap, _ = map_base.create_base_map(sites_file=self.sites_file,
                                           edges_file=self.edges_file)
        self.assertEqual(fmap.base_info["links_drawn"], 2)

    def test_M12_no_marker_is_placed_at_a_wrong_location(self):
        fmap, _ = map_base.create_base_map(sites_file=self.sites_file,
                                           edges_file=self.edges_file)
        path = os.path.join(self.tmp.name, "m.html")
        fmap.save(path)
        text = read_html(path)
        for bad in ("[0.0, 0.0]", "1.5533", "110.3592", "-137.5"):
            self.assertNotIn(bad, text)

    def test_M13_all_sites_bad_or_file_missing_gives_a_clear_error(self):
        write_sites(self.sites_file, self.BAD)
        with self.assertRaises(ValueError):
            map_base.create_base_map(sites_file=self.sites_file, edges_file=self.edges_file)
        with self.assertRaises(FileNotFoundError):
            map_base.create_base_map(sites_file=os.path.join(self.tmp.name, "nope.csv"))


@unittest.skipUnless(HAS_FOLIUM, "folium is not installed:  python -m pip install folium")
class TestRouteMap(unittest.TestCase):
    """M14-M24: Emily's route map against the existing find_routes() output."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_map(self, origin, destination, k, name):
        result = routes_for(origin, destination, k)
        out = os.path.join(self.tmp.name, name + ".html")
        path = map_routes.generate_route_map(result, output_path=out,
                                             origin=origin, destination=destination)
        return result, path

    def test_M14_returns_absolute_path_of_an_existing_html_file(self):
        _, path = self.make_map(2000, 3002, 5, "m14")
        self.assertTrue(os.path.isabs(str(path)))
        self.assertTrue(os.path.exists(str(path)))
        self.assertTrue(str(path).endswith(".html"))

    def test_M24_returned_path_is_a_plain_string(self):
        # The agreed interface says "returns the absolute path". A pathlib.Path
        # works with most code, but the GUI (webbrowser.open, Tk) is safer with str.
        _, path = self.make_map(2000, 3002, 1, "m24")
        self.assertIsInstance(path, str)

    def test_M15_top_k_1_highlights_exactly_one_route(self):
        result, path = self.make_map(2000, 3002, 1, "m15")
        text = read_html(path)
        self.assertEqual(len(result["routes"]), 1)
        self.assertEqual(route_numbers(text), {1})
        self.assertEqual(len(route_colours(text)), 1)

    def test_M16_top_k_5_highlights_five_different_colours(self):
        result, path = self.make_map(2000, 3002, 5, "m16")
        text = read_html(path)
        self.assertEqual(len(result["routes"]), 5)
        self.assertEqual(route_numbers(text), {1, 2, 3, 4, 5})
        self.assertEqual(len(route_colours(text)), 5)

    def test_M17_k_2_and_k_3_show_exactly_that_many_routes(self):
        for k in (2, 3):
            result, path = self.make_map(970, 3002, k, "m17_%d" % k)
            text = read_html(path)
            self.assertEqual(route_numbers(text), set(range(1, k + 1)), "k=%d" % k)

    def test_M18_fewer_routes_than_k_does_not_crash(self):
        result, path = self.make_map(970, 3685, 5, "m18")      # neighbouring sites
        text = read_html(path)
        self.assertLessEqual(len(result["routes"]), 5)
        self.assertEqual(route_numbers(text), {r["rank"] for r in result["routes"]})

    def test_M19_origin_and_destination_markers_are_correct(self):
        _, path = self.make_map(2000, 3002, 5, "m19")
        text = read_html(path)
        self.assertIn("Origin: site 2000", text)
        self.assertIn("Destination: site 3002", text)
        self.assertEqual(text.count('"markerColor": "green"'), 1)
        self.assertEqual(text.count('"markerColor": "red"'), 1)
        _, coords = map_base.create_base_map()
        self.assertIn(str(coords[2000][0]), text)
        self.assertIn(str(coords[3002][0]), text)

    def test_M20_route_sequence_matches_find_routes(self):
        for origin, destination in [(2000, 3002), (970, 3002), (970, 2200)]:
            result, path = self.make_map(origin, destination, 5, "m20_%d_%d" % (origin, destination))
            text = read_html(path)
            for route in result["routes"]:
                sequence = " -> ".join(str(s) for s in route["sites"])
                self.assertIn(sequence, text,
                              "route %d %s -> %s differs from find_routes()"
                              % (route["rank"], origin, destination))

    def test_M21_travel_time_and_distance_match_find_routes(self):
        for origin, destination in [(2000, 3002), (970, 3002), (970, 2200)]:
            result, path = self.make_map(origin, destination, 5, "m21_%d_%d" % (origin, destination))
            text = read_html(path)
            for r in result["routes"]:
                line = "Route %d: %.1f min, %.2f km" % (r["rank"], r["minutes"], r["distance_km"])
                self.assertIn(line, text)
            # the same numbers the terminal / GUI text shows
            for line in tr.format_routes(result).splitlines():
                m = re.match(r"Route (\d+): ([\d.]+) min, ([\d.]+) km", line)
                if m:
                    self.assertIn("Route %s: %s min, %s km" % m.groups(), text)

    def test_M22_special_cases_do_not_crash(self):
        # same origin and destination
        same = routes_for(2000, 2000, 5)
        p1 = map_routes.generate_route_map(same, os.path.join(self.tmp.name, "same.html"),
                                           2000, 2000)
        self.assertTrue(os.path.exists(str(p1)))
        # no routes at all (e.g. sites not connected): the two sites are still marked
        empty = {"routes": [], "info": {"message": "no route: the two sites are not connected"}}
        p2 = map_routes.generate_route_map(empty, os.path.join(self.tmp.name, "empty.html"),
                                           2000, 3002)
        text = read_html(p2)
        self.assertIn("no route", text)
        self.assertIn("Origin: site 2000", text)
        # nothing at all
        p3 = map_routes.generate_route_map({"routes": [], "info": {}},
                                           os.path.join(self.tmp.name, "nothing.html"))
        self.assertTrue(os.path.exists(str(p3)))

    def test_M23_route_through_a_site_without_valid_coordinate_is_safe(self):
        # a route that passes site 4040 whose coordinate is invalid (0, 0)
        tmp_sites = os.path.join(self.tmp.name, "sites_bad.csv")
        sites = pd.read_csv(map_base.SITES_FILE)
        sites.loc[sites["site"] == 4040, ["lat", "lon"]] = [0.0, 0.0]
        sites.to_csv(tmp_sites, index=False)
        real = map_base.create_base_map
        patched = lambda *a, **kw: real(sites_file=tmp_sites)
        result = routes_for(2000, 3002, 5)                     # route 1 uses site 4040
        self.assertIn(4040, result["routes"][0]["sites"])
        with mock.patch.object(map_routes, "create_base_map", patched):
            path = map_routes.generate_route_map(
                result, os.path.join(self.tmp.name, "bad_site.html"), 2000, 3002)
        text = read_html(path)
        self.assertNotIn("[0.0, 0.0]", text)                   # no marker or line to (0, 0)
        self.assertNotIn("Destination: site 4040", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)