#!/usr/bin/env python3
"""
Mission Planning Coursework - Single drone mission

Features included:
- TSP ordering:
    * Euclidean (fast)
    * A*-weighted (slow unless using heuristic + symmetry)  ✅
- Motion planning per leg:
    * Visibility-graph + manual A* (collision-aware) ✅
- Metrics logging:
    * time, commanded distance, detours, avg speed ✅
- CSV mission log output ✅
- ArUco detection verification + image saving per viewpoint ✅

Key fix:
- If USE_ASTAR_WEIGHTED_TSP=True, we compute the TSP order BEFORE takeoff
  so the drone doesn't hover "stuck" while planning.

Run:
  python3 mission_scenario.py -s scenarios/scenario1.yaml
"""

__authors__ = "Rafael Perez-Segui + edits by Oljen"
__license__ = "BSD-3-Clause"

import argparse
from time import sleep
import time
import yaml
import os
import csv
import math
import heapq
from typing import Dict, List, Tuple, Optional

import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

from as2_python_api.drone_interface import DroneInterface

# -----------------------
# CONFIG / FLAGS
# -----------------------
USE_ASTAR_WEIGHTED_TSP = True          # <-- set True for A*-weighted TSP
USE_TSP_HEURISTIC = True              # <-- True recommended when A*-weighted (fast)
TAKE_OFF_HEIGHT = 1.0
TAKE_OFF_SPEED = 1.0
SLEEP_TIME = 0.5
SPEED = 1.0
LAND_SPEED = 0.5

OBSTACLE_MARGIN = 0.6                 # inflate obstacles in planning
CORNER_CLEARANCE = 0.2                # push visibility nodes away from obstacle edges
CRUISE_Z_MODE = "max"                 # "max" -> max(current, goal), or "goal"

OUTPUT_DIR = "outputs"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "mission_log.csv")

# ROS image topic (you confirmed this exists)
DEFAULT_IMAGE_TOPIC = "/drone0/sensor_measurements/hd_camera/image_raw"

# -----------------------
# Metrics store
# -----------------------
METRICS = {"distance_m": 0.0, "detours": 0}


def add_leg_distance(a, b):
    METRICS["distance_m"] += float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


# -----------------------
# Image grabber for ArUco
# -----------------------
class ImageGrabber(Node):
    def __init__(self, topic: str):
        super().__init__("image_grabber_node")
        self._bridge = CvBridge()
        self._latest_bgr = None
        self._sub = self.create_subscription(Image, topic, self._cb, 10)

    def _cb(self, msg: Image):
        self._latest_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def grab_fresh(self, timeout_s: float = 2.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_bgr is not None:
                return self._latest_bgr
        return None


def detect_aruco_ids(bgr) -> List[int]:
    if bgr is None:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [int(x) for x in ids.flatten().tolist()]


# -----------------------
# Obstacle geometry + visibility graph
# -----------------------
def rects_from_obstacles(obstacles: dict, margin: float = OBSTACLE_MARGIN):
    rects = []
    for _, ob in (obstacles or {}).items():
        cx, cy, cz = ob["x"], ob["y"], ob["z"]
        hx = ob["w"] / 2.0 + margin
        hy = ob["d"] / 2.0 + margin
        hz = ob["h"] / 2.0 + margin
        rects.append((cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz))
    return rects


def segment_intersects_rect(p0, p1, rect) -> bool:
    xmin, xmax, ymin, ymax, *_ = rect
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    # Liang–Barsky clip test
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
    for r in rects:
        if segment_intersects_rect(p, q, r):
            return False
    return True


def rect_corners(rect, corner_clearance: float = CORNER_CLEARANCE):
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
    nodes = [start_xy, goal_xy]
    for r in rects:
        nodes.extend(rect_corners(r))

    n = len(nodes)
    nbrs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if visible(nodes[i], nodes[j], rects):
                w = euclid2(nodes[i], nodes[j])
                nbrs[i].append((j, w))
                nbrs[j].append((i, w))
    return nodes, nbrs


def astar_graph(nodes, nbrs, start_idx: int, goal_idx: int):
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
                              margin: float = OBSTACLE_MARGIN,
                              cruise_z: Optional[float] = None):
    sx, sy, sz = start_xyz
    gx, gy, gz = goal_xyz

    rects = rects_from_obstacles(obstacles, margin=margin)
    start_xy = (sx, sy)
    goal_xy = (gx, gy)

    z = cruise_z if cruise_z is not None else gz

    if visible(start_xy, goal_xy, rects):
        return [[gx, gy, z]]

    nodes, nbrs = build_visibility_graph(start_xy, goal_xy, rects)
    path_idx = astar_graph(nodes, nbrs, start_idx=0, goal_idx=1)

    if path_idx is None:
        return [[gx, gy, z]]

    waypoints = []
    for k in path_idx[1:]:
        x, y = nodes[k]
        waypoints.append([x, y, z])

    waypoints[-1][0] = gx
    waypoints[-1][1] = gy
    return waypoints


def planned_path_length(start_xyz, goal_xyz, obstacles) -> float:
    cruise_z = max(start_xyz[2], goal_xyz[2])
    wps = plan_path_visibility_astar(start_xyz, goal_xyz, obstacles, margin=OBSTACLE_MARGIN, cruise_z=cruise_z)
    pts = [start_xyz] + wps
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        total += float(np.linalg.norm(np.array(a) - np.array(b)))
    return total


# -----------------------
# TSP ordering
# -----------------------
def compute_tsp_order_euclid(viewpoint_poses: dict) -> List[int]:
    vp_ids = sorted(viewpoint_poses.keys())
    pts = np.array([[viewpoint_poses[i]["x"], viewpoint_poses[i]["y"], viewpoint_poses[i]["z"]] for i in vp_ids], dtype=float)
    n = len(vp_ids)

    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(pts[i] - pts[j])

    perm, _ = solve_tsp_dynamic_programming(dist)
    return [vp_ids[k] for k in perm]


def compute_tsp_order_astar_cost(viewpoint_poses: dict, drone_start_pose: dict, obstacles: dict) -> List[int]:
    """
    A*-weighted TSP:
    - dist(i,j) = planned collision-free path length between viewpoints i and j
    Optimisations:
    - compute only i<j then mirror (symmetric)
    - optional heuristic TSP solver (recommended)
    """
    vp_ids = sorted(viewpoint_poses.keys())
    n = len(vp_ids)

    print(f"Computing A*-weighted distance matrix for {n} viewpoints (symmetric, ~{n*(n-1)//2} A* legs)...")

    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        if i % 5 == 0:
            print(f"  Progress: row {i+1}/{n}")
        for j in range(i + 1, n):
            vi = viewpoint_poses[vp_ids[i]]
            vj = viewpoint_poses[vp_ids[j]]
            a = [vi["x"], vi["y"], vi["z"]]
            b = [vj["x"], vj["y"], vj["z"]]
            c = planned_path_length(a, b, obstacles)
            dist[i, j] = c
            dist[j, i] = c

    if USE_TSP_HEURISTIC:
        perm, _ = solve_tsp_local_search(dist)
    else:
        perm, _ = solve_tsp_dynamic_programming(dist)

    return [vp_ids[k] for k in perm]


# -----------------------
# Drone mission behaviors
# -----------------------
def drone_start(drone_interface: DroneInterface) -> bool:
    print("Start mission")

    print("Arm")
    success = drone_interface.arm()
    print(f"Arm success: {success}")

    print("Offboard")
    success = drone_interface.offboard()
    print(f"Offboard success: {success}")

    print("Take Off")
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f"Take Off success: {success}")
    return success


def go_to_safe(drone_interface: DroneInterface, start_xyz, goal_vp, obstacles: dict) -> bool:
    sx, sy, sz = start_xyz
    gx, gy, gz = goal_vp["x"], goal_vp["y"], goal_vp["z"]
    goal_xyz = [gx, gy, gz]

    cruise_z = gz if CRUISE_Z_MODE == "goal" else max(sz, gz)

    waypoints = plan_path_visibility_astar(
        start_xyz=[sx, sy, sz],
        goal_xyz=goal_xyz,
        obstacles=obstacles,
        margin=OBSTACLE_MARGIN,
        cruise_z=cruise_z,
    )

    if len(waypoints) > 1:
        METRICS["detours"] += 1

    cur = [sx, sy, sz]
    for wp in waypoints[:-1]:
        add_leg_distance(cur, wp)
        ok = drone_interface.go_to.go_to_point(wp, speed=SPEED)
        if not ok:
            return False
        cur = wp

    final = [gx, gy, gz]
    add_leg_distance(cur, final)
    return drone_interface.go_to.go_to_point_with_yaw(final, angle=goal_vp["w"], speed=SPEED)


def write_csv_log(rows: List[dict], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def drone_run(drone_interface: DroneInterface, scenario: dict, scenario_path: str, image_grabber: ImageGrabber, order: List[int]) -> bool:
    print("Run mission")
    print("Visit order:", order)

    viewpoint_poses = scenario["viewpoint_poses"]
    obstacles = scenario.get("obstacles", {})

    os.makedirs(IMAGES_DIR, exist_ok=True)
    rows = []
    visited_ids = set()

    current = [scenario["drone_start_pose"]["x"], scenario["drone_start_pose"]["y"], TAKE_OFF_HEIGHT]

    for idx, vpid in enumerate(order, start=1):
        vp = viewpoint_poses[vpid]
        print(f"Go to {vpid} with path facing {vp}")

        success = go_to_safe(drone_interface, current, vp, obstacles)
        print(f"Go to success: {success}")
        if not success:
            return False

        current = [vp["x"], vp["y"], vp["z"]]
        print("Go to done")

        bgr = image_grabber.grab_fresh(timeout_s=2.0)
        ids = detect_aruco_ids(bgr)
        ok = len(ids) > 0
        for _id in ids:
            visited_ids.add(_id)

        print(f"Aruco detected at vp {vpid}: {ok}, ids={ids}")

        stamp = int(time.time() * 1000)
        img_name = f"{os.path.basename(scenario_path).replace('.yaml','')}_vp{vpid}_{stamp}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        if bgr is not None:
            cv2.imwrite(img_path, bgr)

        rows.append({
            "scenario": os.path.basename(scenario_path),
            "visit_index": idx,
            "viewpoint_id": vpid,
            "vp_x": vp["x"],
            "vp_y": vp["y"],
            "vp_z": vp["z"],
            "aruco_detected": ok,
            "aruco_ids": str(ids),
            "image_path": img_path,
            "distance_m_so_far": METRICS["distance_m"],
            "detours_so_far": METRICS["detours"],
            "timestamp_s": time.time(),
        })

        sleep(SLEEP_TIME)

    write_csv_log(rows, CSV_LOG_PATH)
    print(f"Log saved to: {CSV_LOG_PATH}")
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"Unique ArUco IDs seen: {sorted(list(visited_ids))}")

    return True


def drone_end(drone_interface: DroneInterface) -> bool:
    print("End mission")

    print("Land")
    success = drone_interface.land(speed=LAND_SPEED)
    print(f"Land success: {success}")
    if not success:
        return success

    print("Manual")
    success = drone_interface.manual()
    print(f"Manual success: {success}")
    return success


def read_scenario(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single drone mission")
    parser.add_argument("-s", "--scenario", type=str, required=True, help="scenario YAML file")
    parser.add_argument("-n", "--namespace", type=str, default="drone0", help="Drone namespace")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("--image_topic", type=str, default=DEFAULT_IMAGE_TOPIC)

    args = parser.parse_args()
    scenario = read_scenario(args.scenario)

    viewpoint_poses = scenario["viewpoint_poses"]
    obstacles = scenario.get("obstacles", {})

    # Compute TSP order BEFORE takeoff (so drone doesn't hover while planning)
    if USE_ASTAR_WEIGHTED_TSP:
        order = compute_tsp_order_astar_cost(viewpoint_poses, scenario["drone_start_pose"], obstacles)
        print("TSP visit order (A*-weighted):", order)
    else:
        order = compute_tsp_order_euclid(viewpoint_poses)
        print("TSP visit order (Euclid):", order)

    rclpy.init()

    uav = DroneInterface(
        drone_id=args.namespace,
        use_sim_time=True,
        verbose=args.verbose,
    )

    image_grabber = ImageGrabber(args.image_topic)

    success = drone_start(uav)

    try:
        METRICS["distance_m"] = 0.0
        METRICS["detours"] = 0

        start_time = time.time()
        if success:
            success = drone_run(uav, scenario, args.scenario, image_grabber, order)
        duration = time.time() - start_time

        print("---------------------------------")
        print(f"Tour of {args.scenario} took {duration} seconds")
        print(f"Distance (commanded): {METRICS['distance_m']:.2f} m")
        print(f"Detours used: {METRICS['detours']}")
        if duration > 0:
            print(f"Avg speed: {METRICS['distance_m'] / duration:.2f} m/s")
        print("---------------------------------")

    except KeyboardInterrupt:
        pass

    drone_end(uav)

    uav.shutdown()
    image_grabber.destroy_node()
    rclpy.shutdown()
    print("Clean exit")
