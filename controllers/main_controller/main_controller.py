from controller import Supervisor
from py_trees.behaviour import Behaviour
from py_trees.common import Status, ParallelPolicy
from py_trees.composites import Sequence,Parallel,Selector
from py_trees import logging as log_tree

import numpy as np
import os

from servoarm import ServoArm
from mapping import Mapping, DoesMapExist
from navigation import Navigation
from planning import Planning
from planning_bfs import PlanningBFS
from planning_simple import PlanningSimple
from object_manipulation import DetectJamJar
from turn_degrees import TurnDegrees
from drive_backward import DriveBackward
from init_object_manip import InitObjectManip
from move_marker import MoveMarker
from approach import Approach

# create the Robot instance.
robot = Supervisor()

#The mapping waypoints. These are the waypoints for the first robot circuit around the table to detect objects inte room
mapping_waypoints = [(0.595, -0.544), (0.595,-2.58), (-0.621, -3.3),(-1.72, -2.46),(-1.72, -2.16),(-1.72, -1.96), (-1.72, -0.431),(-0.416, 0.428),(-1.24, 0.0458),(-1.59, -0.305),(-1.67, -0.651),(-1.67, -1.049),(-1.67, -2.46),(-0.621, -3.3), (0.595, -2.58),(0.595, -0.544),(-0.207,0.263),(-0.207,0.263)]

# Blackboard dictionary. Used to store global state. I don't use a class because a simple dictionary is sufficient.
blackboard = {}
blackboard['robot'] = robot
blackboard['map_width'] = 430 
blackboard['map_height'] = 567 
blackboard['timestep'] = int(robot.getBasicTimeStep())
blackboard['compass'] = robot.getDevice('compass')
blackboard['gps'] = robot.getDevice('gps')
blackboard['lidar'] = robot.getDevice('Hokuyo URG-04LX-UG01')
blackboard['leftmotor'] = robot.getDevice('wheel_left_joint')
blackboard['rightmotor'] = robot.getDevice('wheel_right_joint')
blackboard['camera'] = robot.getDevice('camera')

blackboard.get('compass').enable(blackboard.get('timestep'))
blackboard.get('gps').enable(blackboard.get('timestep'))
blackboard.get('lidar').enable(blackboard.get('timestep'))
blackboard.get('camera').enable(blackboard.get('timestep'))
blackboard.get('camera').recognitionEnable(blackboard.get('timestep'))

#Used to save the cmap
blackboard['filepath'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/cspace.npy'

print(f'world timestep is {blackboard.get('timestep')}')


'''
Below are all of the non-trivial kinematic position presets I use to move the robot's elbow, shoulders, and fingers in addition to the lift joint.
These are packaged up in dictionaries for reuse.
'''
safety = {
    'torso_lift_joint' : 0.35,
    'arm_1_joint' : 0.71,
    'arm_2_joint' : 1.02,
    'arm_3_joint' : -2.815,
    'arm_4_joint' : 1.011,
    'arm_5_joint' : 0,
    'arm_6_joint' : 0,
    'arm_7_joint' : 0,
    'gripper_left_finger_joint' : 0,
    'gripper_right_finger_joint': 0,
    'head_1_joint':0,
    'head_2_joint':0
}

reach = {
    'torso_lift_joint' : 0.26,
    'arm_1_joint' : 1.68,
    'arm_2_joint' : -0.03,
    'arm_3_joint' : -1.6,
    'arm_4_joint' : 0,
    'arm_5_joint' : 0,
    'arm_6_joint' : 0,
    'arm_7_joint' : 0,
    'gripper_left_finger_joint' : 0.045,
    'gripper_right_finger_joint': 0.045,
    'head_1_joint':0,
    'head_2_joint':0	
}

lift_torso = {
    'torso_lift_joint' : 0.35	
}

drop_torso = {
	'torso_lift_joint' : 0.20
}

zero_torso = {
	'torso_lift_joint' : 0.0
}


prep_release = {
	'torso_lift_joint' : 0.20,
}

reach_maintain_grip = {
    'torso_lift_joint' : 0.26,
    'arm_1_joint' : 1.68,
    'arm_2_joint' : -0.03,
    'arm_3_joint' : -1.6,
    'arm_4_joint' : 0,
    'arm_5_joint' : 0,
    'arm_6_joint' : 0,
    'arm_7_joint' : 0
}

bend = {
    'arm_1_joint' : 0.07,
    'arm_2_joint' : 0.004,
    'arm_3_joint' : -1.592,
    'arm_4_joint' : 1.524,
    'arm_5_joint' : 0,
    'arm_6_joint' : 0,
    'arm_7_joint' : 0,
    'head_1_joint':0,
    'head_2_joint':0	
}

bend_left = {
    'arm_1_joint' : 1.5,
    'arm_2_joint' : 0.004,
    'arm_3_joint' : -1.592,
    'arm_4_joint' : 1.524,
    'arm_5_joint' : 0,
    'arm_6_joint' : 0,
    'arm_7_joint' : 0,
    'head_1_joint':0,
    'head_2_joint':0	
}

close_grip = {
    'gripper_left_finger_joint' : 0.035,
    'gripper_right_finger_joint': 0.035,
    'torso_lift_joint' : 0.35   
}

open_grip = {
    'gripper_left_finger_joint' : 0.045,
    'gripper_right_finger_joint': 0.045
}

blackboard['waypoints'] = mapping_waypoints
blackboard['joint_targets'] = safety


'''
Explaination of behavior classes.

DoesMapExist
If the Map Exists, we skip the mapping subroutine and go straight to planning the lower left corner path.

Mapping
The mapping class is standard and uses the same codebase as our previous peer graded assignments.
As input, the Mapping class uses the lidar information to create a probability map and then uses
the completed probability map to produce the convolution map.

PlanningBFS
Planning class uses the BFS algorithm to find the shortest path using 8 way navigation

Planning
I also experimented with Rapidly-exploring Random Trees algorithm (RRT) with straight line collision checking.
My point sampling algorithm biases towards the goal 10% of the time.
As input, the RTT algorithm  uses the convolution map, the robot's current x,y coords, and the goal x,y coords to plan the path.
This algorithm is not as predictable as BFS so I omitted it from the final behavior tree.

ServoArm
Moves the arm joints and the lift joint of the robot. Used for implementing non-trivial kinematics such as bending at the elbow
to move back and forth, using the shoulder joint to steer the arm toward the target

Navigation
Tha Navigation class is also pretty standard. The navigation routine takes as input an array of waypoints, and uses those waypoints to guide the robot on a path from start to goal nodes.
'''
'''
def create_jar_handling_subtree(index, jar_waypoint, blackboard, table_coord):
    return Sequence(
        name=f"Get Jar 1",
        memory=True,
        children=[
            Planning(f"Compute path to sink 1", blackboard, (0.6, 0.704)),
            Navigation(f"Move to sink {index+1}", blackboard),
            ServoArm('Reach for jar 3',reach,blackboard),
            Align(f"Align to jar 1", blackboard),
            ServoArm('Grip Jar 3',close_grip,blackboard)
            TurnDegrees('Turn to place jar 3',blackboard,180),
            Planning(f"Compute path to table ", blackboard, (-0.228, -0.318)),
            Navigation(f"Move to table {index+1}", blackboard),
            ServoArm('Release Jar 3', open_grip, blackboard),
            ServoArm('Lift torso', lift_torso, blackboard),
        ]
    )
'''
jar_positions = [(0.6, 0.704), (0.8, 0.494), (0.8, 0.024)]
blackboard['cspace'] = np.load(blackboard.get('filepath'))

tree = Sequence('Main', children = [
    ServoArm('Move arm to safety', safety, blackboard),
    Sequence(name=f"Get Jar 1", memory=True, children=[
            PlanningBFS(f"Compute path to jar 1", blackboard, (0.6, 0.704)),
            Navigation(f"Move to jar 1", blackboard),
            ServoArm('Reach for jar 3',reach,blackboard),
            Approach(f"Align to jar 1", blackboard),
            ServoArm('Grip Jar 3',close_grip,blackboard),
            TurnDegrees('Turn to place jar 3',blackboard,180),
            PlanningBFS(f"Compute path to table ", blackboard, (-0.228, -0.318)),
            Navigation(f"Move to table 1", blackboard),
            ServoArm('Release Jar 3', open_grip, blackboard),
            ServoArm('Lift torso', lift_torso, blackboard)
    ])
],memory=True)
        

#Initialize the behavior tree and start the main loop
tree.setup_with_descendants()
log_tree.level = log_tree.Level.DEBUG

while robot.step(blackboard.get('timestep')) != -1:
	tree.tick_once()
	if tree.status == Status.SUCCESS:
		print("Simulation complete!")
		break
	elif tree.status == Status.RUNNING:
		print("Simulation running...If the simulation crashes, please restart the Webots app.")