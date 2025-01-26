from controller import Robot,Supervisor
from py_trees.behaviour import Behaviour
from py_trees.common import Status
import numpy as np
import math
from collections import deque

class PlanningSimple(Behaviour):
    def __init__(self,name,waypoints,blackboard):
        super(PlanningSimple, self).__init__(name)
        self.waypoints = waypoints
        self.blackboard = blackboard
        
    def update(self):
        return Status.SUCCESS

    def terminate(self,new_status):
            self.logger.debug(f'PlanningSimple::terminate {self.name}')
            self.blackboard['waypoints'] = self.waypoints
            
            return new_status


