from scipy import signal
from controller import Robot,Supervisor
from matplotlib import pyplot as plt
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.common import ParallelPolicy
from py_trees.composites import Sequence,Parallel,Selector
from py_trees import logging as log_tree
import numpy as np
import math
import os
from os.path import exists
from mapping import Mapping
from navigation import Navigation
from planning import Planning

#The mapping  waypoints. These are skipped when the map exists.
WP = [(0.595, -0.544), (0.595,-2.58), (-0.621, -3.3),(-1.72, -2.46),(-1.72, -2.16),(-1.72, -1.96), (-1.72, -0.431),(-0.416, 0.428),(-1.24, 0.0458),(-1.59, -0.305),(-1.67, -0.651),(-1.67, -1.049),(-1.67, -2.46),(-0.621, -3.3), (0.595, -2.58),(0.595, -0.544),(-0.207,0.263),(-0.207,0.263)]

robot = Supervisor()

#I didn't use a blackboard class. A simple dictionary will suffice for passing data from BT  node to node.
blackboard = {}
blackboard['robot'] = robot
blackboard['waypoints'] = WP
blackboard['filepath'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/cspace.npy'
blackboard['is_map_drawn'] = False
blackboard['map_width'] = 430
blackboard['map_height'] = 567
blackboard['timestep'] = 16 
blackboard['num_navigations'] = 0
'''
py-trees behavior class that checks if saved map exists.
If the map does exist, I load the map and save the convolution space to the blackboard for planning.
If the map does not exist, emit a falure and start the Mapping process.  
'''

class DoesMapExist(Behaviour):
    def update(self):
        file_exists = exists(blackboard.get('filepath'))
        if(file_exists):
            print("Map already exists")
            cspace = np.load(blackboard.get('filepath'))
            blackboard['cspace'] = cspace
            return Status.SUCCESS
        else:
            print("Map does not exist")
            return Status.FAILURE      

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
tree = Sequence("Main", children=[
    Selector("Does map exist?", children=[
    DoesMapExist("Test for map"),
    Parallel("Mapping",ParallelPolicy.SuccessOnOne(), children=[
        Mapping("map the environment", blackboard),
        Navigation("move around the table", blackboard) 
    ])
    ],memory=True),
    Planning("compute path to lower left corner",blackboard,(-1.36,-3.22)),
    #Planning("compute path to lower left corner",blackboard,(0.894,-0.163)),
    Navigation("move to the lower left ocrner",blackboard),
    Planning("compute path to sink",blackboard,(0.01, 0.01)), 
    Navigation("move to sink",blackboard)
],memory=True)

tree.setup_with_descendants()
timestep = blackboard.get('timestep')

while robot.step(timestep) != -1 and blackboard.get('num_navigations') <= 3:
    log_tree.level = log_tree.Level.ERROR
    tree.tick_once()

