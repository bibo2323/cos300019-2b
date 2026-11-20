"""
map_routes.py - COS30019 Assignment 2B (TBRGS) - Research: map visualisation
Owner: Emily Sheryllina Aren Standly

"""
import argparse
import html
import sys
from pathlib import Path

import folium

from map_base import create_base_map

DEFAULT_OUTPUT = "results/route_map.html"

# one colour per route, route 1 first
ROUTE_COLOURS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
ORIGIN_COLOUR = "green"
DESTINATION_COLOUR = "red"


def _segments(sites, coords):
    """Split a route into pieces of [lat, lon] points.
    A site with no valid coordinate breaks the line instead of crashing."""
    # a site with no coordinate cuts the line here instead of crashing
    pieces, current = [], []
    for site in sites:
        point = coords.get(site)
        if point is None:
            if len(current) > 1:
                pieces.append(current)
            current = []
        else:
            current.append(list(point))
    if len(current) > 1:
        pieces.append(current)
    return pieces


def _add_route(fmap, route, colour, weight, coords):

    # each route is its own layer so it can be switched on and off
    label = "Route %d: %.1f min, %.2f km" % (route["rank"], route["minutes"],
                                             route["distance_km"])
    layer = folium.FeatureGroup(name=label, show=True)
    for piece in _segments(route["sites"], coords):
        folium.PolyLine(piece, color=colour, weight=weight, opacity=0.85,
                        tooltip=label + "<br>" + " -> ".join(map(str, route["sites"]))
                        ).add_to(layer)
    layer.add_to(fmap)


def _add_end_marker(fmap, site, coords, colour, title):
    point = coords.get(site)
    if point is None:                      # no valid coordinate, nothing to mark
        return
    folium.Marker(point, tooltip="%s: site %d" % (title, site),
                  icon=folium.Icon(color=colour)).add_to(fmap)


def _summary_box(route_result, origin, destination):
    # the small box in the corner: trip details and one line per route
    info = route_result.get("info", {})
    lines = ["<b>Site %s to site %s</b>" % (origin, destination),
             "Model: %s | Algorithm: %s" % (info.get("model", "-"), info.get("algorithm", "-")),
             "Time used: %s" % info.get("time_used", "-")]
    for i, route in enumerate(route_result.get("routes", [])):
        colour = ROUTE_COLOURS[i % len(ROUTE_COLOURS)]
        lines.append('<span style="color:%s">&#9632;</span> Route %d: %.1f min, %.2f km'
                     % (colour, route["rank"], route["minutes"], route["distance_km"]))
    if info.get("message"):
        lines.append("<i>%s</i>" % html.escape(str(info["message"])))
    return ('<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; '
            'background: white; padding: 8px 12px; border: 1px solid #999; '
            'border-radius: 6px; font: 13px Arial, sans-serif; line-height: 1.5;">'
            + "<br>".join(lines) + "</div>")


def generate_route_map(route_result, output_path=DEFAULT_OUTPUT,
                       origin=None, destination=None):
    """Draw the routes of find_routes() on the base map and save it as HTML.

    route_result : the dict returned by tbrgs_routing.find_routes()
    output_path  : where to save the HTML file
    origin, destination : only needed when there is no route (e.g. sites not
                   connected), so the two sites can still be marked
    Returns the absolute path of the HTML file."""
    fmap, coords = create_base_map()   # Siti's base map and the site coordinates
    routes = route_result.get("routes", [])

    if routes:   # origin and destination come from route 1
        origin, destination = routes[0]["sites"][0], routes[0]["sites"][-1]

    # route 1 is widest, so the others still show where they share a road
    for i, route in enumerate(routes):
        _add_route(fmap, route, ROUTE_COLOURS[i % len(ROUTE_COLOURS)],
                   weight=10 - 1.5 * i, coords=coords)

    if origin is not None:
        _add_end_marker(fmap, int(origin), coords, ORIGIN_COLOUR, "Origin")
    if destination is not None:
        _add_end_marker(fmap, int(destination), coords, DESTINATION_COLOUR, "Destination")

    fmap.get_root().html.add_child(folium.Element(
        _summary_box(route_result, origin, destination)))

    # move Siti's layer switcher to the end, or it does not see the route layers
    for key, child in list(fmap._children.items()):
        if isinstance(child, folium.LayerControl):
            del fmap._children[key]
            fmap.add_child(child)

    # zoom in to the routes, not the whole area
    points = [coords[s] for r in routes for s in r["sites"] if s in coords]
    if len(points) > 1:
        fmap.fit_bounds([[min(p[0] for p in points), min(p[1] for p in points)],
                         [max(p[0] for p in points), max(p[1] for p in points)]])

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(path))
    return str(path.resolve())   # full path as text, so the GUI can open it


def main():
    from tbrgs_routing import find_routes, format_routes   # only needed here
    ap = argparse.ArgumentParser(description="Draw the top-k routes on a map")
    ap.add_argument("origin", type=int)
    ap.add_argument("destination", type=int)
    ap.add_argument("--when", default=None, help='e.g. "2006-10-27 08:00"')
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="gru")
    ap.add_argument("--algorithm", default="AS")
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    result = find_routes(args.origin, args.destination, args.when, args.k,
                         args.model, algorithm=args.algorithm)
    print(format_routes(result))
    path = generate_route_map(result, args.out, args.origin, args.destination)
    print("Map saved to %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
