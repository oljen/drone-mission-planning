# Drone Mission Planning (COMP0240)

Working solution for structural inspection mission planning:
- TSP viewpoint ordering (python_tsp)
- Obstacle-aware safe navigation (AABB footprint intersection + detour waypoint + altitude hop)
- ArUco verification at each viewpoint + logs (CSV) + saved images

## How to run
Terminal 1:
./launch_as2.bash -s scenarios/scenarioX.yaml

Terminal 2:
./launch_ground_station.bash

Terminal 3:
source /opt/ros/humble/setup.bash
source ~/mission_planning_ws/install/setup.bash
python3 mission_scenario.py -s scenarios/scenarioX.yaml
