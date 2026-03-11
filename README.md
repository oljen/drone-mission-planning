# Structural Inspection Path Planning – How to Run

This project implements an autonomous drone inspection planner using:

- Global TSP viewpoint ordering
- Obstacle-aware A* local planning
- Waypoint shortcutting
- ArUco verification
- Mission logging and plotting

The system must be run using **THREE TERMINALS**.

If you do not follow the exact order below, it will not work.

---

# STEP 0 — Open 3 terminals

You will run:

Terminal 1 → Simulation  
Terminal 2 → Ground station  
Terminal 3 → Mission script  

---

# STEP 1 — Run the Simulation

In **Terminal 1**, run:

```bash
source /opt/ros/humble/setup.bash
source ~/mission_planning_ws/install/setup.bash

cd ~/mission_planning_ws/src/challenge_mission_planning
./launch_as2.bash -s scenarios/scenario2.yaml

# STEP 2

In **Terminal 2**, run:

source /opt/ros/humble/setup.bash
source ~/mission_planning_ws/install/setup.bash

cd ~/mission_planning_ws/src/challenge_mission_planning
./launch_ground_station.bash


# STEP 3

source /opt/ros/humble/setup.bash
source ~/mission_planning_ws/install/setup.bash

cd ~/mission_planning_ws/src/challenge_mission_planning
python3 mission_scenario.py -s scenarios/scenario2.yaml
