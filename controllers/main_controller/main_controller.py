from controller import Supervisor
from py_trees.behaviour import Behaviour
from py_trees.common import Status, ParallelPolicy
from py_trees.composites import Sequence,Parallel,Selector
from py_trees import logging as log_tree

import numpy as np

from servoarm import ServoArm
from mapping import Mapping
from navigation import Navigation

# create the Robot instance.
robot = Supervisor()

#The mapping  waypoints.
mapping_waypoints = [(0.595, -0.544), (0.595,-2.58), (-0.621, -3.3),(-1.72, -2.46),(-1.72, -2.16),(-1.72, -1.96), (-1.72, -0.431),(-0.416, 0.428),(-1.24, 0.0458),(-1.59, -0.305),(-1.67, -0.651),(-1.67, -1.049),(-1.67, -2.46),(-0.621, -3.3), (0.595, -2.58),(0.595, -0.544),(-0.207,0.263),(-0.207,0.263)]

# Used to store global state
blackboard = {}
blackboard['robot'] = robot
blackboard['map_width'] = 430 
blackboard['map_height'] = 567 
#blackboard['timestep'] = int(robot.getBasicTimeStep())
blackboard['timestep'] = int(robot.getBasicTimeStep())
blackboard['compass'] = robot.getDevice('compass')
blackboard['gps'] = robot.getDevice('gps')
blackboard['lidar'] = robot.getDevice('Hokuyo URG-04LX-UG01')
blackboard['leftmotor'] = robot.getDevice('wheel_left_joint')
blackboard['rightmotor'] = robot.getDevice('wheel_right_joint')

blackboard.get('compass').enable(blackboard.get('timestep'))
blackboard.get('gps').enable(blackboard.get('timestep'))
blackboard.get('lidar').enable(blackboard.get('timestep'))


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

blackboard['waypoints'] = mapping_waypoints

# Behavior tree for robot behavior sequencing
tree = Sequence("Main", children = [
	ServoArm('Move arm to safety',safety,blackboard),
	Parallel("Mapping",ParallelPolicy.SuccessOnOne(), children=[
        Mapping("map the environment", blackboard),
        Navigation("move around the table", blackboard) 
    ])
],memory=True)


tree.setup_with_descendants()
log_tree.level = log_tree.Level.DEBUG
while robot.step(blackboard.get('timestep')) != -1:
	tree.tick_once()
	'''if tree.status == Status.SUCCESS:
		print("All joints reached their target positions.")
		break
	elif tree.status == Status.RUNNING:
		print("Moving joints to target positions...")'''