from py_trees.behaviour import Behaviour
from py_trees.common import Status

class MoveMarker(Behaviour):
    def __init__(self, name, blackboard):

        super(MoveMarker, self).__init__(name)
        self.robot = blackboard.get('robot')
        self.blackboard = blackboard
        self.marker = self.robot.getFromDef('marker').getField('translation')

        self.has_run = False
    def update(self):
        if not self.has_run:
            self.marker.setSFVec3f([-0.56,0.46,0])
            return Status.SUCCESS
        