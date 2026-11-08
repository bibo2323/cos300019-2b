"""
travel_time.py - COS30019 Assignment 2B (TBRGS)

Converts traffic flow into travel time for ONE edge (road link) between two
SCATS sites, following "Traffic Flow to Travel Time Conversion v1.0":

    flow = -1.4648375 * speed^2 + 93.75 * speed        (flow in vehicles/hour)

Steps:
    1. flow  -> speed   (capped at the speed limit, 60 km/h)
    2. time  = distance / speed  + intersection delay (30 seconds)

Usage from other files:
    from travel_time import edge_travel_time_seconds, haversine_km
    seconds = edge_travel_time_seconds(distance_km=1.2, flow_per_hour=600)
"""
import math

# ---- assumptions from the brief / conversion PDF (change here only) --------
SPEED_LIMIT_KMH = 60.0          # same speed limit on every link
INTERSECTION_DELAY_S = 30.0     # average delay to pass a controlled intersection
A = -1.4648375                  # flow = A*speed^2 + B*speed
B = 93.75
CAPACITY_FLOW = B * B / (-4.0 * A)      # = 1500 vehicles/hour (top of the curve)
CAPACITY_SPEED = B / (-2.0 * A)         # = 32 km/h (speed when flow = capacity)
EARTH_RADIUS_KM = 6371.0
# -----------------------------------------------------------------------------


def flow_to_speed(flow_per_hour, speed_limit=SPEED_LIMIT_KMH, congested=False):
    """Speed (km/h) for a given flow (vehicles/hour).

    congested=False -> green line (road under capacity): the normal case.
    congested=True  -> red line (road over capacity): the slow branch.

    Flow above CAPACITY_FLOW (1500) has no solution on the curve, so it is
    treated as exactly capacity (speed = 32 km/h). Our data goes above 1500,
    so this rule is an assumption we must state in the report."""
    flow = max(0.0, float(flow_per_hour))
    flow = min(flow, CAPACITY_FLOW)

    # solve  A*s^2 + B*s - flow = 0  (quadratic formula)
    disc = max(0.0, B * B + 4.0 * A * flow)      # never negative after the clamp
    root = math.sqrt(disc)
    s_under = (-B - root) / (2.0 * A)            # green line (higher speed)
    s_over = (-B + root) / (2.0 * A)             # red line (lower speed)
    speed = s_over if congested else s_under
    return min(speed, speed_limit)               # cannot exceed the speed limit


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two latitude/longitude points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def edge_travel_time_seconds(distance_km, flow_per_hour,
                             delay_s=INTERSECTION_DELAY_S):
    """Travel time in SECONDS for one edge = distance/speed + intersection delay."""
    speed = flow_to_speed(flow_per_hour)
    return distance_km / speed * 3600.0 + delay_s


def min_edge_travel_time_seconds(distance_km, delay_s=INTERSECTION_DELAY_S):
    """Fastest possible time for an edge (free flow, 60 km/h). Used as the A*
    heuristic: it never overestimates, so A* stays correct."""
    return distance_km / SPEED_LIMIT_KMH * 3600.0 + delay_s


def _self_test():
    """Quick checks so we can prove the formula works (also good for the report)."""
    print("capacity: %.0f veh/h at %.1f km/h" % (CAPACITY_FLOW, CAPACITY_SPEED))
    for flow in (0, 200, 351, 352, 600, 1000, 1500, 3000):
        speed = flow_to_speed(flow)
        t = edge_travel_time_seconds(1.0, flow)
        print("flow %4d veh/h -> speed %5.1f km/h -> 1 km edge takes %5.1f s"
              % (flow, speed, t))
    # speed at 351 veh/h must be (almost) the speed limit
    assert abs(flow_to_speed(351) - 60.0) < 0.2
    assert abs(flow_to_speed(1500) - 32.0) < 0.1
    assert edge_travel_time_seconds(1.0, 100) == 60.0 + INTERSECTION_DELAY_S
    print("All checks passed.")


if __name__ == "__main__":
    _self_test()
