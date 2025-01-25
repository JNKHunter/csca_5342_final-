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
# create the Robot instance.
robot = Supervisor()

#The mapping  waypoints.
mapping_waypoints = [(0.595, -0.544), (0.595,-2.58), (-0.621, -3.3),(-1.72, -2.46),(-1.72, -2.16),(-1.72, -1.96), (-1.72, -0.431),(-0.416, 0.428),(-1.24, 0.0458),(-1.59, -0.305),(-1.67, -0.651),(-1.67, -1.049),(-1.67, -2.46),(-0.621, -3.3), (0.595, -2.58),(0.595, -0.544),(-0.207,0.263),(-0.207,0.263)]
jar1_waypoints = [(0.957,-0.082)]
jar1_place_waypoints = [(0.38,-0.583)]
jar2_waypoints = [(1.14,0.466)]


# Used to store global state
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

blackboard['filepath'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/cspace.npy'

print(f'world timestep is {blackboard.get('timestep')}')

#Data
#Robot arm safe position
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
    'arm_1_joint' : 2.6,
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

blackboard['waypoints'] = jar1_waypoints
blackboard['joint_targets'] = safety


'''
The behavior tree declaration.
DoesMapExist
If the Map Exists, we skip the mapping subroutine and go straight to planning the lower left corner path.

Mapping
The mapping class is standard and uses the same codebase as our previous peer graded assignments.
As input, the Mapping class uses the lidar information to create a probability map and then uses
the completed probability map to produce the convolution map.

Planning
Planning class uses the Rapidly-exploring Random Trees algorithm (RRT) with straight line collision checking.
My point sampling algorithm biases towards the goal 10% of the time.
As input, the RTT algorithm  uses the convolution map, the robot's current x,y coords, and the goal x,y coords to plan the path.

Navigation
Tha Navigation class is also pretty standard. The navigation routine takes as input an array of waypoints, and uses those waypoints to guide the robot on a path from start to goal nodes.
'''

'''
tree = Sequence("Main", children = [
	ServoArm('Move arm to safety',safety,blackboard),
	Selector('Does map exist?', children=[
        DoesMapExist('Check for saved map',blackboard),
        Parallel("Mapping",ParallelPolicy.SuccessOnOne(), children=[
            Mapping("map the environment", blackboard),
            Navigation("move around the table", blackboard) 
        ])		
    ],memory=True),
	Planning("compute path to lower left corner",blackboard,(-1.36,-3.22)),
	Navigation("move to the lower left ocrner",blackboard),
    Planning("compute path to sink",blackboard,(0.01, 0.01)), 
    Navigation("move to sink",blackboard)
],memory=True)
'''

tree = Sequence('Main', children = [
	#ServoArm('Move arm to safety',safety,blackboard),
    #DetectJamJar('Detect Jars', blackboard),
    ServoArm('Move arm to Jar 1', reach, blackboard),
		Selector('Does map exist?', children=[
        DoesMapExist('Check for saved map',blackboard),
        Parallel("Mapping",ParallelPolicy.SuccessOnOne(), children=[
            Mapping("map the environment", blackboard),
            Navigation("move around the table", blackboard) 
        ])		
    ],memory=True),
	Sequence('Place all 3 jars', children = [
		Sequence('Jar 1', children = [
            PlanningSimple("Path to Jar 1",jar1_waypoints,blackboard),
            Navigation('Move robot to Jar 1',blackboard),
            ServoArm('Grip Jar 1',close_grip,blackboard),
            ServoArm('Bend Arm',bend,blackboard),
            PlanningSimple("Path to place Jar 1", [(0.38,-0.583)],blackboard),
            Navigation('Move robot to place Jar 1',blackboard),
            ServoArm('Move arm to place Jar 1', reach_maintain_grip, blackboard),
            ServoArm('Release Jar 1', open_grip, blackboard)			
        ],memory=True),
		Sequence('Jar 2', children = [
            ServoArm('Bend Arm',bend,blackboard),
            PlanningSimple('Plan turn towards Jar 2', [(0.707,0.0141)],blackboard),
            Navigation('Turn to Jar 2',blackboard),
            ServoArm('Move arm to Jar 2', reach, blackboard),
            PlanningSimple('Path towards Jar 2', [(1.09,0.22)],blackboard),
            Navigation('move robot to Jar 2',blackboard),
            ServoArm('Grip Jar 2',close_grip,blackboard),
            ServoArm('Bend Arm',bend,blackboard),
            TurnDegrees('Turn 180 jar 2',blackboard,180),
            ServoArm('Move arm to place Jar 2', reach_maintain_grip, blackboard),
            PlanningSimple('Path towards Jar 2', [(0.208,-0.212)],blackboard),
            Navigation('move robot to place Jar 2',blackboard),
			ServoArm('Prep Release Jar 2', prep_release, blackboard),
            ServoArm('Release Jar 2', open_grip, blackboard)			
        ],memory=True),
		Sequence('Jar 3', children = [
            ServoArm('Bend Arm',bend,blackboard),
			ServoArm('Bend Arm',safety,blackboard),
            TurnDegrees('Turn towards jar 2',blackboard,135),
			ServoArm('Bend Arm',bend,blackboard),
			ServoArm('Reach for jar 3',reach,blackboard),
			PlanningSimple('Path towards Jar 3', [(1.27,0.170)],blackboard),
			#1.13,0.241
			Navigation('Move robot to Jar 3',blackboard)
        ],memory=True)
    ],memory=True)
],memory=True)
#TODO after turn 180 for jar 3, the robot seems unstable
tree.setup_with_descendants()
log_tree.level = log_tree.Level.DEBUG

while robot.step(blackboard.get('timestep')) != -1:
	tree.tick_once()
	if tree.status == Status.SUCCESS:
		print("All joints reached their target positions.")
		break
	elif tree.status == Status.RUNNING:
		print("Moving joints to target positions...")