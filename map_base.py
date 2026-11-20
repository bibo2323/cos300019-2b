"""
map_base.py - COS30019 Assignment 2B (TBRGS) - Research: map visualisation
"""
import os
import sys

import folium
import pandas as pd

# ---- settings (change here only) -------------------------------------------
SITES_FILE = os.path.join("data", "processed", "sites.csv")
EDGES_FILE = os.path.join("data", "processed", "edges.csv")
DEFAULT_OUTPUT = os.path.join("results", "base_map.html")

# A coordinate is valid only if it is inside Boroondara's surroundings. This
# also removes (0, 0), the bad value found in the raw data for site 4266.
LAT_RANGE = (-37.95, -37.70)
LON_RANGE = (144.95, 145.20)

# The brief warns that SCATS coordinates may not sit exactly on the real
# intersections. If the markers look shifted when you open the map, adjust
# these two numbers (degrees; 0.0001 is about 10 metres) and re-run.
LAT_OFFSET = 0.0
LON_OFFSET = 0.0

# Background tiles. The free OpenStreetMap tile server BLOCKS maps that are opened
# as a file on your computer (you see "403 Access blocked"), so the default is a
# street map from Esri that has no such rule. The map also has a layer switcher
# (top right) so you can change the background if one of them does not load.
TILE_LAYERS = [   # (name, url, attribution, subdomains)
    ("Esri street map",
     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
     "Tiles &copy; Esri", "abc"),
    ("CartoDB light (OpenStreetMap data)",
     "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
     "&copy; OpenStreetMap contributors &copy; CARTO", "abcd"),
    ("OpenStreetMap (only works from a web server)",
     "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
     "&copy; OpenStreetMap contributors", "abc"),
]

DEFAULT_ZOOM = 13
SITE_COLOUR = "#1f77b4"            # blue: connected site
ISOLATED_COLOUR = "#ff7f0e"        # orange: site with no road link
ROAD_COLOUR = "#808080"            # grey road network
# -----------------------------------------------------------------------------


def is_valid_coordinate(lat, lon):
    """True if lat/lon are real numbers inside the allowed Boroondara box.
    Missing values (NaN/None), text and (0, 0) all return False."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if lat != lat or lon != lon:                       # NaN check
        return False
    return (LAT_RANGE[0] <= lat <= LAT_RANGE[1]
            and LON_RANGE[0] <= lon <= LON_RANGE[1])


def load_site_coordinates(sites_file=SITES_FILE):
    """Read sites.csv. Returns (coords, names, skipped):
        coords  {site: (lat, lon)}   only valid sites (offset already applied)
        names   {site: location name}
        skipped list of site numbers dropped because of bad coordinates."""
    if not os.path.exists(sites_file):
        raise FileNotFoundError(
            "Cannot find %s. Run  python data_processing.py  first." % sites_file)
    sites = pd.read_csv(sites_file)
    coords, names, skipped = {}, {}, []
    for row in sites.itertuples(index=False):
        site = int(row.site)
        if is_valid_coordinate(row.lat, row.lon):
            coords[site] = (float(row.lat) + LAT_OFFSET, float(row.lon) + LON_OFFSET)
            names[site] = str(row.name)
        else:
            skipped.append(site)
    return coords, names, skipped


def load_edges(edges_file=EDGES_FILE):
    """Read edges.csv as a DataFrame (empty one if the file does not exist)."""
    if not os.path.exists(edges_file):
        return pd.DataFrame(columns=["a", "b", "distance_km", "road", "kind"])
    return pd.read_csv(edges_file)


def create_base_map(zoom_start=DEFAULT_ZOOM, sites_file=SITES_FILE,
                    edges_file=EDGES_FILE):
    """Build the base map.

    Returns (folium_map, coords) where coords = {site: (lat, lon)} for every
    site that has a valid coordinate."""
    coords, names, skipped = load_site_coordinates(sites_file)
    if not coords:
        raise ValueError("No SCATS site has a valid coordinate.")

    # centre the map on the average position of the valid sites
    centre = (sum(c[0] for c in coords.values()) / len(coords),
              sum(c[1] for c in coords.values()) / len(coords))
    fmap = folium.Map(location=centre, zoom_start=zoom_start, tiles=None)
    for i, (name, url, attr, subdomains) in enumerate(TILE_LAYERS):
        folium.TileLayer(tiles=url, attr=attr, name=name, subdomains=subdomains,
                         show=(i == 0), max_zoom=19).add_to(fmap)

    # --- road network (grey lines), drawn first so markers sit on top -------
    edges = load_edges(edges_file)
    linked = set()
    drawn = 0
    for row in edges.itertuples(index=False):
        a, b = int(row.a), int(row.b)
        if a not in coords or b not in coords:          # skip links with a bad end
            continue
        folium.PolyLine(
            [coords[a], coords[b]], color=ROAD_COLOUR, weight=3, opacity=0.8,
            tooltip="%d - %d | %s | %.2f km" % (a, b, row.road, row.distance_km),
        ).add_to(fmap)
        linked.update((a, b))
        drawn += 1

    # --- SCATS sites (markers) -----------------------------------------------
    for site, (lat, lon) in coords.items():
        isolated = site not in linked
        colour = ISOLATED_COLOUR if isolated else SITE_COLOUR
        note = "isolated: no road link" if isolated else "connected"
        folium.CircleMarker(
            [lat, lon], radius=6, color=colour, weight=2,
            fill=True, fill_color=colour, fill_opacity=0.9,
            tooltip="Site %d" % site,
            popup=folium.Popup("<b>Site %d</b><br>%s<br>%s"
                               % (site, names[site], note), max_width=250),
        ).add_to(fmap)

    # layer switcher for the background (Emily: do NOT add a second LayerControl)
    folium.LayerControl(collapsed=True).add_to(fmap)

    # fit the view to all valid sites
    lats = [c[0] for c in coords.values()]
    lons = [c[1] for c in coords.values()]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    # keep a short summary on the object so tests / the GUI can read it
    fmap.base_info = {"sites_drawn": len(coords), "links_drawn": drawn,
                      "skipped_sites": skipped}
    return fmap, coords


def save_base_map(path=DEFAULT_OUTPUT, **kwargs):
    """Create the base map and save it as an HTML file.
    Returns the absolute path of the file."""
    fmap, _ = create_base_map(**kwargs)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fmap.save(path)
    return os.path.abspath(path)


def main():
    fmap, coords = create_base_map()
    info = fmap.base_info
    path = save_base_map()
    print("Base map created.")
    print("  sites drawn   : %d" % info["sites_drawn"])
    print("  links drawn   : %d" % info["links_drawn"])
    print("  skipped sites : %s" % (info["skipped_sites"] or "none"))
    print("Saved to %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
