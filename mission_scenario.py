#!/usr/bin/env python3
"""Optimised mission: TSP tour + obstacle-aware detours + ArUco verification + CSV logging."""

__authors__ = 'Rafael Perez-Segui'
__copyright__ = 'Copyright (c) 2024 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

import argparse
from time import sleep
import time
import yaml
import heapq
import math


import os
import csv
from datetime import datetime

import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from as2_python_api.drone_interface import DroneInterface

import cv2
from cv2 import aruco
from cv_bridge import CvBridge


TAKE_OFF_HEIGHT = 1.0  # Height in meters
TAKE_OFF_SPEED = 1.0   # Max speed in m/s
SLEEP_TIME = 0.5       # Sleep time between behaviors in seconds
SPEED = 1.0            # Max speed in m/s
LAND_SPEED = 0.5       # Max speed in m/s

# ✅ Confirmed from your ros2 topic list
CAMERA_TOPIC = "/drone0/sensor_measurements/hd_camera/image_raw"

OUTPUT_DIR = "outputs"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LOG_PATH = os.path.join(OUTPUT_DIR, "mission_log.csv")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Simple metrics store (accumulates during mission run)
METRICS = {
    "distance_m": 0.0,
    "detours": 0,
}


def add_leg_distance(a, b):
    """Add Euclidean distance between 3D points a and b to METRICS."""
    METRICS["distance_m"] += float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def init_log():
    """Create CSV log file with header if missing."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_utc", "scenario_path", "scenario_name",
                "viewpoint_id", "x", "y", "z", "yaw_w",
                "go_to_success", "detours_total_so_far",
                "aruco_detected", "aruco_ids",
                "image_path"
            ])


def append_log_row(scenario_path, scenario_name, vpid, vp, go_to_success, aruco_detected, aruco_ids, image_path):
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.utcnow().isoformat(),
            scenario_path,
            scenario_name,
            int(vpid),
            float(vp["x"]), float(vp["y"]), float(vp["z"]), float(vp["w"]),
            bool(go_to_success),
            int(METRICS["detours"]),
            bool(aruco_detected),
            ";".join(map(str, aruco_ids)),
            image_path
        ])


class ImageGrabber(Node):
    """
    Keep a subscription open and allow 'grab_latest()' to fetch the latest received frame.
    We reset 'latest' before each grab so we know it is fresh.
    """
    def __init__(self, topic: str):
        super().__init__("image_grabber")
        self._bridge = CvBridge()
        self._latest_bgr = None
        self._sub = self.create_subscription(Image, topic, self._cb, 10)

    def _cb(self, msg: Image):
        try:
            self._latest_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")

    def grab_fresh(self, timeout_s: float = 2.0):
        """
        Clear latest image, then spin until we receive a new one or timeout.
        """
        self._latest_bgr = None
        t0 = time.time()
        while self._latest_bgr is None and (time.time() - t0) < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._latest_bgr


def detect_aruco_ids(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # A bit of contrast helps in sim
    gray = cv2.equalizeHist(gray)

    # Try a bunch of common ArUco + AprilTag dictionaries
    dict_candidates = [
        # ArUco
        getattr(aruco, "DICT_4X4_50", None),
        getattr(aruco, "DICT_4X4_100", None),
        getattr(aruco, "DICT_4X4_250", None),
        getattr(aruco, "DICT_5X5_50", None),
        getattr(aruco, "DICT_5X5_100", None),
        getattr(aruco, "DICT_5X5_250", None),
        getattr(aruco, "DICT_6X6_50", None),
        getattr(aruco, "DICT_6X6_100", None),
        getattr(aruco, "DICT_6X6_250", None),
        getattr(aruco, "DICT_ARUCO_ORIGINAL", None),

        # AprilTag (OpenCV 4.7+ usually)
        getattr(aruco, "DICT_APRILTAG_16h5", None),
        getattr(aruco, "DICT_APRILTAG_25h9", None),
        getattr(aruco, "DICT_APRILTAG_36h10", None),
        getattr(aruco, "DICT_APRILTAG_36h11", None),
    ]
    dict_candidates = [d for d in dict_candidates if d is not None]

    params = aruco.DetectorParameters()

    # Make detector a bit more forgiving (helps with perspective / blur)
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0

    for d in dict_candidates:
        aruco_dict = aruco.getPredefinedDictionary(d)
        detector = aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            return True, [int(x) for x in ids.flatten().tolist()]

    return False, []


def drone_start(drone_interface: DroneInterface) -> bool:
    """Take off the drone."""
    print('Start mission')

    print('Arm')
    success = drone_interface.arm()
    print(f'Arm success: {success}')

    print('Offboard')
    success = drone_interface.offboard()
    print(f'Offboard success: {success}')

    print('Take Off')
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f'Take Off success: {success}')

    return success


def compute_tsp_order(viewpoint_poses: dict) -> list[int]:
    """Compute TSP order using Euclidean distance between viewpoints."""
    vp_ids = sorted(viewpoint_poses.keys())
    pts = np.array([[viewpoint_poses[i]["x"], viewpoint_poses[i]["y"], viewpoint_poses[i]["z"]] for i in vp_ids], dtype=float)

    n = len(vp_ids)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(pts[i] - pts[j])

    perm, _ = solve_tsp_dynamic_programming(dist)
    return [vp_ids[k] for k in perm]


def obstacle_aabb(ob: dict, margin: float = 0.6):
    """
    Axis-aligned bounding box (AABB) for obstacle with safety margin.
    Returns (xmin, xmax, ymin, ymax, zmin, zmax).
    """
    cx, cy, cz = ob["x"], ob["y"], ob["z"]
    hx = ob["w"] / 2.0 + margin
    hy = ob["d"] / 2.0 + margin
    hz = ob["h"] / 2.0 + margin
    return (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)


def segment_intersects_aabb_2d(p0, p1, aabb) -> bool:
    """Check if 2D segment p0->p1 intersects the obstacle rectangle in XY (Liang–Barsky)."""
    xmin, xmax, ymin, ymax, *_ = aabb

    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return True


def compute_detour_waypoint(p0, p1, aabb, clearance: float = 1.0):
    """
    Create a detour waypoint around the obstacle in XY.
    Try 4 corner waypoints around the inflated AABB and choose a valid + shortest.
    """
    xmin, xmax, ymin, ymax, *_ = aabb

    candidates = [
        (xmin - clearance, ymin - clearance),
        (xmin - clearance, ymax + clearance),
        (xmax + clearance, ymin - clearance),
        (xmax + clearance, ymax + clearance),
    ]

    valid = []
    for c in candidates:
        if (not segment_intersects_aabb_2d(p0, c, aabb)) and (not segment_intersects_aabb_2d(c, p1, aabb)):
            valid.append(c)

    if not valid:
        valid = candidates

    def path_cost(c):
        return ((p0[0] - c[0])**2 + (p0[1] - c[1])**2) ** 0.5 + ((p1[0] - c[0])**2 + (p1[1] - c[1])**2) ** 0.5

    return min(valid, key=path_cost)


def rects_from_obstacles(obstacles: dict, margin: float = 0.6):
    """
    Convert obstacles dict into list of inflated axis-aligned rectangles in XY.
    Returns list of (xmin, xmax, ymin, ymax, zmin, zmax).
    """
    rects = []
    for _, ob in (obstacles or {}).items():
        cx, cy, cz = ob["x"], ob["y"], ob["z"]
        hx = ob["w"] / 2.0 + margin
        hy = ob["d"] / 2.0 + margin
        hz = ob["h"] / 2.0 + margin
        rects.append((cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz))
    return rects


def segment_intersects_rect(p0, p1, rect) -> bool:
    """
    Liang–Barsky clip test for intersection between segment p0->p1 and axis-aligned rectangle.
    rect = (xmin, xmax, ymin, ymax, zmin, zmax) but we use only XY.
    Returns True if segment intersects rectangle interior/boundary.
    """
    xmin, xmax, ymin, ymax, *_ = rect
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return True


def visible(p, q, rects) -> bool:
    """True if straight segment p->q does NOT intersect any rectangle in rects."""
    for r in rects:
        if segment_intersects_rect(p, q, r):
            return False
    return True


def rect_corners(rect, corner_clearance: float = 0.2):
    """
    Return 4 corner points (x,y) for inflated rectangle, with a tiny extra clearance
    so we don't sit exactly on the boundary.
    """
    xmin, xmax, ymin, ymax, *_ = rect
    return [
        (xmin - corner_clearance, ymin - corner_clearance),
        (xmin - corner_clearance, ymax + corner_clearance),
        (xmax + corner_clearance, ymin - corner_clearance),
        (xmax + corner_clearance, ymax + corner_clearance),
    ]


def euclid2(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_visibility_graph(start_xy, goal_xy, rects):
    """
    Build visibility graph: nodes are start, goal, and rectangle corners.
    Returns:
      nodes: list[(x,y)]
      nbrs: adjacency list, nbrs[i] = list[(j, weight)]
    """
    nodes = [start_xy, goal_xy]
    for r in rects:
        nodes.extend(rect_corners(r))

    n = len(nodes)
    nbrs = [[] for _ in range(n)]

    # Connect if visible; O(n^2 * #rects) but small n (good for coursework)
    for i in range(n):
        for j in range(i + 1, n):
            if visible(nodes[i], nodes[j], rects):
                w = euclid2(nodes[i], nodes[j])
                nbrs[i].append((j, w))
                nbrs[j].append((i, w))
    return nodes, nbrs


def astar_graph(nodes, nbrs, start_idx: int, goal_idx: int):
    """
    Manual A* on a weighted graph.
    nodes: list[(x,y)]
    nbrs: adjacency list
    Returns list of node indices representing path, or None if no path.
    """
    def h(i):
        return euclid2(nodes[i], nodes[goal_idx])

    open_heap = []
    heapq.heappush(open_heap, (h(start_idx), 0.0, start_idx))

    came_from = {}
    gscore = {start_idx: 0.0}
    closed = set()

    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal_idx:
            # Reconstruct
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        closed.add(cur)

        for nxt, w in nbrs[cur]:
            tentative = g + w
            if nxt in closed:
                continue
            if tentative < gscore.get(nxt, float("inf")):
                gscore[nxt] = tentative
                came_from[nxt] = cur
                heapq.heappush(open_heap, (tentative + h(nxt), tentative, nxt))

    return None


def plan_path_visibility_astar(start_xyz, goal_xyz, obstacles: dict,
                              margin: float = 0.6,
                              cruise_z: float = None):
    """
    Plan collision-free path in XY using visibility graph + A*.
    Returns list of XYZ waypoints INCLUDING goal, excluding start.
    """
    sx, sy, sz = start_xyz
    gx, gy, gz = goal_xyz

    rects = rects_from_obstacles(obstacles, margin=margin)

    start_xy = (sx, sy)
    goal_xy = (gx, gy)

    # If straight line is already clear, just go direct
    if visible(start_xy, goal_xy, rects):
        z = cruise_z if cruise_z is not None else gz
        return [[gx, gy, z]]

    nodes, nbrs = build_visibility_graph(start_xy, goal_xy, rects)
    path_idx = astar_graph(nodes, nbrs, start_idx=0, goal_idx=1)

    # Fallback: if graph fails (rare), go direct (your old detour could be used here instead)
    if path_idx is None:
        z = cruise_z if cruise_z is not None else gz
        return [[gx, gy, z]]

    z = cruise_z if cruise_z is not None else gz

    # Convert node indices to waypoints; skip index 0 (start)
    waypoints = []
    for k in path_idx[1:]:
        x, y = nodes[k]
        waypoints.append([x, y, z])

    # Ensure last waypoint is exactly the goal XY (numerical safety)
    waypoints[-1][0] = gx
    waypoints[-1][1] = gy
    return waypoints


def go_to_safe(drone_interface: DroneInterface, start_xyz, goal_vp, obstacles: dict) -> bool:
    """
    Uses visibility graph + A* to produce a collision-free XY path.
    Flies intermediate waypoints using go_to_point, then final approach with yaw.
    """
    sx, sy, sz = start_xyz
    gx, gy, gz = goal_vp["x"], goal_vp["y"], goal_vp["z"]
    goal_xyz = [gx, gy, gz]

    # Cruise altitude: keep current or use goal altitude (pick one)
    cruise_z = max(sz, gz)  # simple and stable

    waypoints = plan_path_visibility_astar(
        start_xyz=[sx, sy, sz],
        goal_xyz=goal_xyz,
        obstacles=obstacles,
        margin=0.6,
        cruise_z=cruise_z
    )

    # Detour metric: if planner gave >1 waypoint, it had to go around something
    if len(waypoints) > 1:
        METRICS["detours"] += 1

    # Fly intermediate waypoints (excluding final goal, which we do with yaw)
    cur = [sx, sy, sz]
    for wp in waypoints[:-1]:
        add_leg_distance(cur, wp)
        ok = drone_interface.go_to.go_to_point(wp, speed=SPEED)
        if not ok:
            return False
        cur = wp

    # Final with yaw at real goal altitude
    final = [gx, gy, gz]
    add_leg_distance(cur, final)
    return drone_interface.go_to.go_to_point_with_yaw(final, angle=goal_vp["w"], speed=SPEED)


def drone_run(
    drone_interface: DroneInterface,
    scenario: dict,
    scenario_path: str,
    scenario_name: str,
    image_grabber: ImageGrabber,
) -> bool:
    print('Run mission')

    viewpoint_poses = scenario["viewpoint_poses"]
    obstacles = scenario.get("obstacles", {})

    order = compute_tsp_order(viewpoint_poses)
    print("TSP visit order:", order)

    current = [scenario["drone_start_pose"]["x"], scenario["drone_start_pose"]["y"], TAKE_OFF_HEIGHT]

    init_log()

    for vpid in order:
        vp = viewpoint_poses[vpid]
        print(f'Go to {vpid} with path facing {vp}')

        go_ok = go_to_safe(drone_interface, current, vp, obstacles)
        print(f'Go to success: {go_ok}')
        if not go_ok:
            append_log_row(scenario_path, scenario_name, vpid, vp, False, False, [], "")
            return False

        current = [vp["x"], vp["y"], vp["z"]]
        print('Go to done')
        sleep(SLEEP_TIME)

        # -------- ArUco evidence capture + verification --------
        bgr = image_grabber.grab_fresh(timeout_s=2.0)
        img_path = ""
        ar_ok, ar_ids = False, []

        if bgr is not None:
            img_path = os.path.join(IMAGES_DIR, f"{scenario_name}_vp{vpid}.png")
            cv2.imwrite(img_path, bgr)
            ar_ok, ar_ids = detect_aruco_ids(bgr)
            print(f"Aruco detected at vp {vpid}: {ar_ok}, ids={ar_ids}")
        else:
            print(f"[WARN] No camera image received for vp {vpid} (topic: {CAMERA_TOPIC})")

        append_log_row(scenario_path, scenario_name, vpid, vp, True, ar_ok, ar_ids, img_path)

    return True


def drone_end(drone_interface: DroneInterface) -> bool:
    """End the mission for a single drone."""
    print('End mission')

    print('Land')
    success = drone_interface.land(speed=LAND_SPEED)
    print(f'Land success: {success}')
    if not success:
        return success

    print('Manual')
    success = drone_interface.manual()
    print(f'Manual success: {success}')
    return success


def read_scenario(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single drone mission')

    parser.add_argument('scenario', type=str, help="scenario file to attempt to execute")
    parser.add_argument('-n', '--namespace', type=str, default='drone0', help='ID of the drone to be used in the mission')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True, help='Use simulation time')

    args = parser.parse_args()

    drone_namespace = args.namespace
    verbosity = args.verbose
    use_sim_time = args.use_sim_time

    print(f'Running mission for drone {drone_namespace}')
    print(f"Reading scenario {args.scenario}")

    scenario = read_scenario(args.scenario)
    scenario_name = str(scenario.get("name", os.path.splitext(os.path.basename(args.scenario))[0]))

    rclpy.init()

    # Node for camera capture (kept alive throughout mission)
    image_grabber = ImageGrabber(CAMERA_TOPIC)

    uav = DroneInterface(
        drone_id=drone_namespace,
        use_sim_time=use_sim_time,
        verbose=verbosity
    )

    success = drone_start(uav)

    try:
        METRICS["distance_m"] = 0.0
        METRICS["detours"] = 0

        start_time = time.time()
        if success:
            success = drone_run(uav, scenario, args.scenario, scenario_name, image_grabber)
        duration = time.time() - start_time

        print("---------------------------------")
        print(f"Tour of {args.scenario} took {duration} seconds")
        print(f"Distance (commanded): {METRICS['distance_m']:.2f} m")
        print(f"Detours used: {METRICS['detours']}")
        if duration > 0:
            print(f"Avg speed: {METRICS['distance_m'] / duration:.2f} m/s")
        print(f"Log saved to: {LOG_PATH}")
        print(f"Images saved to: {IMAGES_DIR}")
        print("---------------------------------")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            image_grabber.destroy_node()
        except Exception:
            pass

    _ = drone_end(uav)

    uav.shutdown()
    rclpy.shutdown()
    print('Clean exit')
    exit(0)
