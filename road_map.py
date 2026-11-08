"""
road_map.py - COS30019 Assignment 2B (TBRGS)

Builds the road map (graph) of the Boroondara area. Nodes = SCATS sites
(intersections). Edges = road links between neighbouring sites.

The VicRoads files give site names and coordinates but NOT which sites are
connected, so we build the links ourselves (assumptions are listed below):

  Rule 1 (shared road):  each SCATS location name tells us the roads meeting at
          a site, e.g. "WARRIGAL_RD N of HIGH STREET_RD". Sites that share a
          road name are lined up along that road (sorted by position) and each
          site is linked to its next neighbour on that road.
  Rule 2 (isolated sites): a site that shares no road with any other site is
          linked to its 2 nearest sites (marked kind='assumed').
  Every link is two-way (cars can drive both directions).
  Distance = straight-line (haversine) distance between the two sites.

Run once:   python road_map.py        -> writes data/processed/edges.csv
Use:        from road_map import load_edges, build_adjacency
"""
import os
import re
import sys

import numpy as np
import pandas as pd

from data_processing import read_raw, load_sites, OUT_DIR
from travel_time import haversine_km

EDGES_FILE = os.path.join(OUT_DIR, "edges.csv")
NEAREST_LINKS_FOR_ISOLATED = 2          # Rule 2: how many neighbours to add

# "WARRIGAL_RD N of HIGH STREET_RD"  ->  road 1, direction, road 2
_LOCATION = re.compile(r"^(.*?)\s+(N|S|E|W|NE|NW|SE|SW)\s+of\s+(.*)$", re.I)


def _clean_road(name):
    """Tidy a road name so the same road matches across sites."""
    name = name.strip().upper()
    if "EASTERN_FWY" in name:               # ramps and the freeway = one road
        return "EASTERN_FWY"
    return name


def roads_per_site(raw=None):
    """{road name: set of site numbers on that road} from the Location column."""
    raw = read_raw() if raw is None else raw
    roads = {}
    for site, location in raw[["site", "location"]].drop_duplicates().itertuples(index=False):
        match = _LOCATION.match(str(location).strip())
        if not match:
            continue
        for road in (match.group(1), match.group(3)):
            roads.setdefault(_clean_road(road), set()).add(int(site))
    return roads


def build_edges(sites=None, raw=None):
    """Returns a DataFrame with columns: a, b, distance_km, road, kind.
    Each link is listed ONCE (a < b); use build_adjacency() for both directions."""
    sites = load_sites() if sites is None else sites
    pos = sites.set_index("site")[["lat", "lon"]]
    links = {}                                                  # (a, b) -> (road, kind)

    # Rule 1: link neighbours along every road that has 2+ sites
    for road, site_set in roads_per_site(raw).items():
        ids = sorted(s for s in site_set if s in pos.index)
        if len(ids) < 2:
            continue
        coords = pos.loc[ids].to_numpy()
        axis = 0 if np.ptp(coords[:, 0]) >= np.ptp(coords[:, 1]) else 1   # longer direction
        ordered = [ids[i] for i in np.argsort(coords[:, axis])]
        for a, b in zip(ordered, ordered[1:]):
            links.setdefault((min(a, b), max(a, b)), (road, "road"))

    # Rule 2: connect isolated sites to their nearest neighbours
    linked = {s for pair in links for s in pair}
    for site in pos.index:
        if site in linked:
            continue
        dist = {other: haversine_km(pos.loc[site, "lat"], pos.loc[site, "lon"],
                                    pos.loc[other, "lat"], pos.loc[other, "lon"])
                for other in pos.index if other != site}
        for other in sorted(dist, key=dist.get)[:NEAREST_LINKS_FOR_ISOLATED]:
            links.setdefault((min(site, other), max(site, other)), ("nearest", "assumed"))

    rows = []
    for (a, b), (road, kind) in sorted(links.items()):
        d = haversine_km(pos.loc[a, "lat"], pos.loc[a, "lon"],
                         pos.loc[b, "lat"], pos.loc[b, "lon"])
        rows.append((a, b, round(d, 4), road, kind))
    return pd.DataFrame(rows, columns=["a", "b", "distance_km", "road", "kind"])


def save_edges(path=EDGES_FILE):
    edges = build_edges()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    edges.to_csv(path, index=False)
    return edges


def load_edges(path=EDGES_FILE):
    return pd.read_csv(path)


def build_adjacency(edges=None):
    """{site: {neighbour: distance_km}} with BOTH directions of every link.
    This is what the search code (2A) uses to expand a node."""
    edges = load_edges() if edges is None else edges
    adj = {}
    for a, b, d in edges[["a", "b", "distance_km"]].itertuples(index=False):
        adj.setdefault(int(a), {})[int(b)] = float(d)
        adj.setdefault(int(b), {})[int(a)] = float(d)
    return adj


def connected_components(adj):
    """List of sets of sites that can reach each other."""
    seen, parts = set(), []
    for start in adj:
        if start in seen:
            continue
        stack, part = [start], set()
        while stack:
            node = stack.pop()
            if node in part:
                continue
            part.add(node)
            stack.extend(set(adj[node]) - part)
        seen |= part
        parts.append(part)
    return parts


def main():
    edges = save_edges()
    adj = build_adjacency(edges)
    parts = connected_components(adj)
    print("Road map built.")
    print("  sites with links : %d" % len(adj))
    print("  links (two-way)  : %d  (%d from shared roads, %d assumed)"
          % (len(edges), (edges["kind"] == "road").sum(), (edges["kind"] == "assumed").sum()))
    print("  separate parts   : %d" % len(parts))
    print("  longest link     : %.2f km" % edges["distance_km"].max())
    print("Saved to %s" % EDGES_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
