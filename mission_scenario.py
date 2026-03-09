#!/usr/bin/env python3
"""
Mission Planning Coursework - Single drone mission

Planner modes:
- baseline: sequential YAML order
- tsp_euclid: TSP using Euclidean edge weights
- tsp_astar: A*-weighted ordering, with exact open-path optimisation for small scenarios

Features:
- Visibility-graph + manual A* local planner
- Path shortcutting
- Metrics logging: time, commanded distance, detours, avg speed
- Planning-time logging: TSP + A* planning time
- CSV mission log output
- ArUco detection verification + image saving per viewpoint
- Mission completion summary
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
import itertools
from typing import List, Optional, Set

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
USE_TSP_HEURISTIC = True              # recommended for larger problems
EXACT_OPEN_PATH_LIMIT = 8             # if viewpoints <= this, brute-force best start-aware path
TAKE_OFF_HEIGHT = 1.0
TAKE_OFF_SPEED = 1.0
SLEEP_TIME = 0.2                      # reduced a bit for speed
SPEED = 1.0
LAND_SPEED = 0.5

OBSTACLE_MARGIN = 0.6
CORNER_CLEARANCE = 0.2
CRUISE_Z_MODE = "max"                 # "max" or "goal"

# For strict validation, set e.g. {24, 34, 44, 54, 64}
REQUIRED_ARUCO_IDS = None

OUTPUT_DIR = "outputs"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "mission_log.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "mission_summary.txt")

DEFAULT_IMAGE_TOPIC = "/drone0/sensor_measurements/hd_camera/image_raw"

# Faster but still robust scan
ARUCO_SCANS = 3
ARUCO_DT = 0.08
CAMERA_SETTLE_TIME = 0.15

# -----------------------
# Metrics store
# -----------------------
METRICS = {
    "distance_m": 0.0,
    "detours": 0,
    "tsp_planning_time_s": 0.0,
    "astar_planning_time_s": 0.0,
    "astar_calls": 0,
}


def reset_metrics():
    METRICS["distance_m"] = 0.0
    METRICS["detours"] = 0
    METRICS["tsp_planning_time_s"] = 0.0
    METRICS["astar_planning_time_s"] = 0.0
    METRICS["astar_calls"] = 0


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
        self._latest_stamp = None
        self._sub = self.create_subscription(Image, topic, self._cb, 10)

    def _cb(self, msg: Image):
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._latest_bgr = bgr
            self._latest_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")

    def grab_fresh(self, timeout_s: float = 2.0):
        t0 = time.time()
        start_stamp = self._latest_stamp
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_bgr is None:
                continue
            if start_stamp is None or self._latest_stamp != start_stamp:
                return self._latest_bgr
        return self._latest_bgr


def detect_aruco_ids(bgr) -> List[int]:
    if bgr is None:
        return []

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    dict_candidates = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_5X5_100,
        cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_6X6_100,
    ]

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10

    best_ids = []
    for d in dict_candidates:
        aruco_dict = cv2.aruco.getPredefinedDictionary(d)
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) > len(best_ids):
            best_ids = [int(x) for x in ids.flatten().tolist()]

    return best_ids


def scan_aruco_multi_frame(image_grabber: ImageGrabber, scans: int = ARUCO_SCANS, dt: float = ARUCO_DT) -> List[int]:
    seen: Set[int] = set()
    for _ in range(scans):
        bgr = image_grabber.grab_fresh(timeout_s=1.0)
        ids = detect_aruco_ids(bgr)
        for _id in ids:
            seen.add(_id)
        time.sleep(dt)
    return sorted(seen)


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


def shortcut_waypoints(start_xyz, waypoints, rects):
    """
    Remove unnecessary intermediate waypoints if later points are directly visible.
    waypoints excludes start, includes goal.
    """
    if len(waypoints) <= 1:
        return waypoints

    pts = [start_xyz] + waypoints
    simplified = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1:
            if visible((pts[i][0], pts[i][1]), (pts[j][0], pts[j][1]), rects):
                break
            j -= 1
        simplified.append(pts[j])
        i = j

    return simplified[1:]


def plan_path_visibility_astar(start_xyz, goal_xyz, obstacles: dict,
                               margin: float = OBSTACLE_MARGIN,
                               cruise_z: Optional[float] = None):
    t0 = time.time()
    METRICS["astar_calls"] += 1

    sx, sy, sz = start_xyz
    gx, gy, gz = goal_xyz

    rects = rects_from_obstacles(obstacles, margin=margin)
    start_xy = (sx, sy)
    goal_xy = (gx, gy)
    z = cruise_z if cruise_z is not None else gz

    if visible(start_xy, goal_xy, rects):
        METRICS["astar_planning_time_s"] += (time.time() - t0)
        return [[gx, gy, z]]

    nodes, nbrs = build_visibility_graph(start_xy, goal_xy, rects)
    path_idx = astar_graph(nodes, nbrs, start_idx=0, goal_idx=1)

    if path_idx is None:
        METRICS["astar_planning_time_s"] += (time.time() - t0)
        return [[gx, gy, z]]

    waypoints = []
    for k in path_idx[1:]:
        x, y = nodes[k]
        waypoints.append([x, y, z])

    waypoints[-1][0] = gx
    waypoints[-1][1] = gy

    waypoints = shortcut_waypoints(start_xyz, waypoints, rects)

    METRICS["astar_planning_time_s"] += (time.time() - t0)
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
# Ordering
# -----------------------
def compute_baseline_order(viewpoint_poses: dict) -> List[int]:
    return list(sorted(viewpoint_poses.keys()))


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


def compute_best_open_order_exact_astar(viewpoint_poses: dict, drone_start_pose: dict, obstacles: dict) -> List[int]:
    """
    Exact open-path optimisation from the drone start pose.
    For small scenarios only. Objective is:
      start -> v1 -> v2 -> ... -> vn
    with no return-to-start/end cycle requirement.
    """
    t0 = time.time()

    vp_ids = sorted(viewpoint_poses.keys())
    n = len(vp_ids)
    start_xyz = [drone_start_pose["x"], drone_start_pose["y"], TAKE_OFF_HEIGHT]

    # Precompute start->vp and vp->vp costs
    start_cost = {}
    pair_cost = {}

    for i, vpid in enumerate(vp_ids):
        vp = viewpoint_poses[vpid]
        goal = [vp["x"], vp["y"], vp["z"]]
        start_cost[vpid] = planned_path_length(start_xyz, goal, obstacles)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a_id = vp_ids[i]
            b_id = vp_ids[j]
            va = viewpoint_poses[a_id]
            vb = viewpoint_poses[b_id]
            a = [va["x"], va["y"], va["z"]]
            b = [vb["x"], vb["y"], vb["z"]]
            pair_cost[(a_id, b_id)] = planned_path_length(a, b, obstacles)

    best_order = None
    best_cost = float("inf")

    print(f"Computing exact open-path order from start for {n} viewpoints ({math.factorial(n)} permutations)...")

    for perm in itertools.permutations(vp_ids):
        total = start_cost[perm[0]]
        for a, b in zip(perm[:-1], perm[1:]):
            total += pair_cost[(a, b)]

        if total < best_cost:
            best_cost = total
            best_order = list(perm)

    METRICS["tsp_planning_time_s"] = time.time() - t0
    print(f"Best open-path cost from start: {best_cost:.2f}")
    return best_order


def compute_tsp_order_astar_cost(viewpoint_poses: dict, drone_start_pose: dict, obstacles: dict) -> List[int]:
    """
    A*-weighted ordering.
    Uses exact open-path optimisation for small scenarios, otherwise TSP on pairwise A* costs.
    """
    vp_ids = sorted(viewpoint_poses.keys())
    n = len(vp_ids)

    if n <= EXACT_OPEN_PATH_LIMIT:
        return compute_best_open_order_exact_astar(viewpoint_poses, drone_start_pose, obstacles)

    t0 = time.time()

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

    METRICS["tsp_planning_time_s"] = time.time() - t0
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


def write_summary_file(scenario_name: str, mission_complete: bool, visited_ids: Set[int], duration: float):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        f.write(f"Scenario: {scenario_name}\n")
        f.write(f"Mission complete: {mission_complete}\n")
        f.write(f"Detected IDs: {sorted(list(visited_ids))}\n")
        f.write(f"Distance (m): {METRICS['distance_m']:.2f}\n")
        f.write(f"Detours: {METRICS['detours']}\n")
        f.write(f"Avg speed (m/s): {(METRICS['distance_m'] / duration) if duration > 0 else 0.0:.2f}\n")
        f.write(f"TSP planning time (s): {METRICS['tsp_planning_time_s']:.3f}\n")
        f.write(f"A* planning time total (s): {METRICS['astar_planning_time_s']:.3f}\n")
        f.write(f"A* calls: {METRICS['astar_calls']}\n")


def drone_run(drone_interface: DroneInterface, scenario: dict, scenario_path: str,
              image_grabber: ImageGrabber, order: List[int], mode: str):
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
            return False, visited_ids

        current = [vp["x"], vp["y"], vp["z"]]
        print("Go to done")

        time.sleep(CAMERA_SETTLE_TIME)

        ids = scan_aruco_multi_frame(image_grabber, scans=ARUCO_SCANS, dt=ARUCO_DT)
        ok = len(ids) > 0
        for _id in ids:
            visited_ids.add(_id)

        print(f"Aruco detected at vp {vpid}: {ok}, ids={ids}")

        bgr_save = image_grabber.grab_fresh(timeout_s=0.5)

        stamp = int(time.time() * 1000)
        img_name = f"{os.path.basename(scenario_path).replace('.yaml','')}_vp{vpid}_{stamp}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        if bgr_save is not None:
            cv2.imwrite(img_path, bgr_save)

        rows.append({
            "scenario": os.path.basename(scenario_path),
            "mode": mode,
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

    if REQUIRED_ARUCO_IDS is not None:
        missing = set(REQUIRED_ARUCO_IDS) - visited_ids
        mission_complete = len(missing) == 0
        print(f"Mission complete: {mission_complete}")
        print(f"Missing IDs: {sorted(list(missing))}")
    else:
        mission_complete = len(visited_ids) > 0
        print(f"Mission complete (based on any detections): {mission_complete}")

    return mission_complete, visited_ids


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
    parser.add_argument("--mode", type=str, default="tsp_astar",
                        choices=["baseline", "tsp_euclid", "tsp_astar"],
                        help="Planner mode")

    args = parser.parse_args()
    scenario = read_scenario(args.scenario)

    viewpoint_poses = scenario["viewpoint_poses"]
    obstacles = scenario.get("obstacles", {})

    reset_metrics()

    if args.mode == "baseline":
        t0 = time.time()
        order = compute_baseline_order(viewpoint_poses)
        METRICS["tsp_planning_time_s"] = time.time() - t0
        print("Visit order (baseline):", order)
    elif args.mode == "tsp_euclid":
        t0 = time.time()
        order = compute_tsp_order_euclid(viewpoint_poses)
        METRICS["tsp_planning_time_s"] = time.time() - t0
        print("TSP visit order (Euclid):", order)
    else:
        order = compute_tsp_order_astar_cost(viewpoint_poses, scenario["drone_start_pose"], obstacles)
        print("TSP visit order (A*-weighted):", order)

    rclpy.init()

    uav = DroneInterface(
        drone_id=args.namespace,
        use_sim_time=True,
        verbose=args.verbose,
    )

    image_grabber = ImageGrabber(args.image_topic)

    success = drone_start(uav)
    mission_complete = False
    visited_ids = set()

    try:
        start_time = time.time()
        if success:
            mission_complete, visited_ids = drone_run(
                uav, scenario, args.scenario, image_grabber, order, args.mode
            )
        duration = time.time() - start_time

        print("---------------------------------")
        print(f"Mode: {args.mode}")
        print(f"Tour of {args.scenario} took {duration} seconds")
        print(f"Distance (commanded): {METRICS['distance_m']:.2f} m")
        print(f"Detours used: {METRICS['detours']}")
        if duration > 0:
            print(f"Avg speed: {METRICS['distance_m'] / duration:.2f} m/s")
        print(f"TSP planning time: {METRICS['tsp_planning_time_s']:.3f} s")
        print(f"A* planning time total: {METRICS['astar_planning_time_s']:.3f} s")
        print(f"A* calls: {METRICS['astar_calls']}")
        print(f"Mission complete: {mission_complete}")
        print("---------------------------------")

        write_summary_file(os.path.basename(args.scenario), mission_complete, visited_ids, duration)

    except KeyboardInterrupt:
        pass

    drone_end(uav)

    uav.shutdown()
    image_grabber.destroy_node()
    rclpy.shutdown()
    print("Clean exit")
