"""
tbrgs_routing.py - COS30019 Assignment 2B (TBRGS).  Owner: Emily Sheryllina Aren Standly

"""
import argparse
import heapq
import importlib
import math
import sys

import pandas as pd

import data_processing as dp
import road_map as rm
import travel_time as tt
import ts_common as tc
from algorithms import METHODS            # the 6 Part A search methods
from problem import Problem


DEFAULT_MODEL = "gru"            
DEFAULT_K = 5                    #up to five routes
DEFAULT_ALGORITHM = "AS"         # one of DFS, BFS, GBFS, AS, CUS1, CUS2
MAX_NODES = 300000               
FLOW_BASIS = "sum"              
DATA_START = pd.Timestamp("2006-10-01")
DATA_END = pd.Timestamp("2006-11-01")         
LAST_WEEK_START = pd.Timestamp("2006-10-25")   

KM_PER_DEGREE = 6371.0 * math.pi / 180.0                    # length of 1 degree of latitude
SECONDS_PER_KM = 3600.0 / tt.SPEED_LIMIT_KMH                # 60 s per km at 60 km/h

_network = {}                    
_series = {}                    


def load_network():
    """(sites table indexed by site number, adjacency {site: {neighbour: km}})."""
    if not _network:
        _network["sites"] = dp.load_sites().set_index("site")
        _network["adj"] = rm.build_adjacency()
    return _network["sites"], _network["adj"]


def reachable_from(adj, start):
    """All sites that can be reached from `start` (so we can say 'not connected')."""
    seen, stack = {start}, [start]
    while stack:
        here = stack.pop()
        for nxt in adj.get(here, {}):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def to_dataset_time(when=None):
    """Turn any time into a 15-minute time INSIDE the October 2006 data.

    The data only covers October 2006, so a time outside it (including 'now')
    is mapped to the same weekday and same time of day in the last week of the
    data (25-31 Oct). Returns (timestamp, was_mapped)."""
    ts = (pd.Timestamp.now() if when is None else pd.Timestamp(when)).floor("15min")
    if DATA_START <= ts < DATA_END:
        return ts, False
    
    for day in pd.date_range(LAST_WEEK_START, periods=7):
        if day.weekday() == ts.weekday():
            return day + (ts - ts.normalize()), True
    raise ValueError("cannot map %s into the data" % ts)    

#predicted flow of every site 
def _all_series():
    if not _series:
        _series.update(tc.load_series())
    return _series


def historic_flow_per_hour(site, when):
    """Average flow (vehicles/hour) at this site for the same weekday and time
    of day. Used as a fallback, and as the 'historic' no-ML baseline."""
    ser = _all_series()[int(site)]
    idx = ser.index
    at_time = (idx.hour == when.hour) & (idx.minute == when.minute)
    same_day = ser[at_time & (idx.day_of_week == when.weekday())]
    values = same_day if len(same_day) else ser[at_time]
    return 4.0 * float(values.mean())


def predict_flows(model, when, flow_basis=FLOW_BASIS):
    """{site: vehicles/hour} for ALL sites at time `when`.

    `model` is the name of a model file: "gru" uses gru_model.py, "lstm" uses
    lstm_model.py, and so on. Each must offer predict_flow_per_hour(site, when).
    A site with no data in the 3 hours before `when` gets the historic average
    instead. flow_basis: see FLOW_BASIS at the top.
    Returns (flows, sites_that_used_the_fallback)."""
    sites, _ = load_network()
    predictor = None
    if model != "historic":
        predictor = importlib.import_module(model + "_model").predict_flow_per_hour
    flows, fallback = {}, []
    for site in sites.index:
        flow = None
        if predictor is not None:
            try:
                flow = predictor(int(site), when)
            except ValueError:               
                flow = None
        if flow is None:
            flow = historic_flow_per_hour(site, when)
            if predictor is not None:
                fallback.append(int(site))
        if flow_basis == "mean_per_approach":          # flow of one road link, not the whole site
            flow = flow / max(1, int(sites.loc[site, "n_approaches"]))
        flows[int(site)] = float(flow)
    return flows, fallback


#Part A problem with travel times as costs
def build_problem(origin, destination, flows):
    """A Part A Problem object where
         node  = SCATS site
         cost  = travel time in SECONDS of a link A -> B, using the flow at A
         (x,y) = position in km x 60, so the straight-line distance between two
                 nodes equals the time needed at 60 km/h. problem.h() then never
                 overestimates (nothing is faster than 60 km/h), so A* stays correct
                 without touching problem.py."""
    sites, adj = load_network()
    lat0, lon0 = sites["lat"].mean(), sites["lon"].mean()
    # km x 60 = seconds at 60 km/h
    scale = KM_PER_DEGREE * SECONDS_PER_KM
    nodes = {int(s): ((r.lon - lon0) * math.cos(math.radians(lat0)) * scale,
                      (r.lat - lat0) * scale) for s, r in sites.iterrows()}
    edges = {}
    for a, neighbours in adj.items():
        for b, dist_km in neighbours.items():
            edges[(a, b)] = tt.edge_travel_time_seconds(dist_km, flows[a])   
    return Problem(nodes, edges, origin, [destination])


# search: A* that does not stop at the first goal 
def _edges_of(path):
    return set(zip(path, path[1:]))


def top_k_routes(problem, k=DEFAULT_K, max_overlap=None):

    root = problem.root()
    created, counter = 1, 0
    frontier = [(problem.h(root.state), root.state, counter, root)]
    routes = []
    while frontier and len(routes) < k:
        _, _, _, node = heapq.heappop(frontier)
        if problem.is_goal(node.state):        # save the route, keep searching
            path = node.path()
            too_similar = max_overlap is not None and any(
                len(_edges_of(path) & _edges_of(p)) / max(1, len(_edges_of(path))) > max_overlap
                for p, _ in routes)
            if not too_similar:
                routes.append((path, node.g))
            continue
        children = problem.expand(node)
        created += len(children)               # count every child
        if created > MAX_NODES:
            break
        for child in children:
            if node.on_path(child.state):
                continue
            counter += 1
            heapq.heappush(frontier, (child.g + problem.h(child.state),
                                      child.state, counter, child))
    return routes, created


# the function the GUI calls 
def find_routes(origin, destination, when=None, k=DEFAULT_K, model=DEFAULT_MODEL,
                flows=None, max_overlap=None, flow_basis=FLOW_BASIS,
                algorithm=DEFAULT_ALGORITHM):
   
    sites, adj = load_network()
    origin, destination = int(origin), int(destination)
    algorithm = algorithm.upper()
    if algorithm not in METHODS:
        raise ValueError("Unknown algorithm %s (choose from %s)" % (algorithm, ", ".join(METHODS)))
    for site in (origin, destination):
        if site not in adj:
            raise ValueError("Unknown SCATS site %d" % site)

    ts, mapped = to_dataset_time(when)
    info = {"model": model if flows is None else "given flows", "algorithm": algorithm,
            "time_used": ts,
            "time_was_mapped": mapped, "fallback_sites": [], "nodes_created": 0,
            "message": ""}

    if origin == destination:                  # already there
        info["message"] = "origin and destination are the same site"
        return {"routes": [_describe(1, [origin], 0.0, adj)], "info": info}
    if destination not in reachable_from(adj, origin):
        info["message"] = "no route: the two sites are not connected"
        return {"routes": [], "info": info}

    if flows is None:
        flows, info["fallback_sites"] = predict_flows(model, ts, flow_basis)
    problem = build_problem(origin, destination, flows)
    if algorithm == "AS":                      # only A* gives k routes
        found, info["nodes_created"] = top_k_routes(problem, k, max_overlap)
    else:                                      # the other Part A methods return one path
        goal, info["nodes_created"], path = METHODS[algorithm](problem)
        found = [(path, problem.path_cost(path))] if goal is not None else []
    routes = [_describe(i, path, secs, adj) for i, (path, secs) in enumerate(found, 1)]
    if algorithm != "AS":
        info["message"] = "%s returns one route (AS returns up to %d)" % (algorithm, k)
    elif len(routes) < k:
        info["message"] = "only %d route(s) exist" % len(routes)
    return {"routes": routes, "info": info}


def _describe(rank, path, seconds, adj):
    return {"rank": rank, "sites": path, "seconds": seconds, "minutes": seconds / 60.0,
            "distance_km": sum(adj[a][b] for a, b in zip(path, path[1:]))}


def format_routes(result):
    """Plain-text version of find_routes() for the terminal and the report."""
    info = result["info"]
    lines = ["model: %s | algorithm: %s | time used: %s%s" % (
        info["model"], info["algorithm"], info["time_used"],
        " (mapped from the requested time)" if info["time_was_mapped"] else "")]
    if info["fallback_sites"]:
        lines.append("historic average used for sites: %s" % info["fallback_sites"])
    if not result["routes"]:
        lines.append(info["message"])
        return "\n".join(lines)
    for r in result["routes"]:
        lines.append("Route %d: %.1f min, %.2f km: %s" % (
            r["rank"], r["minutes"], r["distance_km"], " -> ".join(str(s) for s in r["sites"])))
    if info["message"]:
        lines.append(info["message"])
    lines.append("nodes created by the search: %d" % info["nodes_created"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Top-k routes between two SCATS sites")
    ap.add_argument("origin", type=int)
    ap.add_argument("destination", type=int)
    ap.add_argument("--when", default=None, help='e.g. "2006-10-27 08:00"; default = now')
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--algorithm", default=DEFAULT_ALGORITHM, type=str.upper, choices=list(METHODS),
                    help="Part A search method (default AS)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="gru, lstm, historic, ...")
    ap.add_argument("--basis", default=FLOW_BASIS, choices=["sum", "mean_per_approach"],
                    help="what counts as the flow of a site (see FLOW_BASIS)")
    ap.add_argument("--different", type=float, default=None,
                    help="0-1: skip routes sharing more than this share of links with a better one")
    args = ap.parse_args()
    result = find_routes(args.origin, args.destination, args.when, args.k, args.model,
                         max_overlap=args.different, flow_basis=args.basis,
                         algorithm=args.algorithm)
    print(format_routes(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
