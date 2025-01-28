from py_trees.behaviour import Behaviour
from py_trees.common import Status
import numpy as np

class Wait(Behaviour):
    def __init__(self, name):
        super(Wait, self).__init__(name)
        self.ticks = 0

    def initialise(self):
        print("pausing for a few ticks")
    
    def update(self):
        if self.ticks < 50:
            self.ticks += 1
            return Status.RUNNING
        else:
            return Status.SUCCESS