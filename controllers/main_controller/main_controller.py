from controller import Supervisor
from py_trees.behaviour import Behaviour
from py_trees.common import Status
import numpy as np
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.common import ParallelPolicy
from py_trees.composites import Sequence,Parallel,Selector

from servoarm import ServoArm

# create the Robot instance.
robot = Supervisor()

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

print(f'world timestep is {blackboard.get('timestep')}')

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

# Behavior tree for robot behavior sequencing
tree = Sequence("Main", children = [ServoArm('Move arm to safety',safety,blackboard)], memory=False)

while robot.step(blackboard.get('timestep')) != -1:
	tree.tick_once()
	if tree.status == Status.SUCCESS:
		print("All joints reached their target positions.")
		break
	elif tree.status == Status.RUNNING:
		print("Moving joints to target positions...")