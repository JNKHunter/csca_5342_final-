from py_trees.behaviour import Behaviour
from py_trees.common import Status

class InitObjectManip(Behaviour):
    def __init__(self, name, blackboard):

        super(InitObjectManip, self).__init__(name)
        self.robot = blackboard.get('robot')
        self.robot_node = self.robot.getSelf()
        self.blackboard = blackboard

        self.has_run = False
    def update(self):
        if not self.has_run:
            translation_field = self.robot_node.getField("translation")
            rotation_field = self.robot_node.getField("rotation")

            # Set new translation (x, y, z)
            new_translation = [0.0, 0.18, 0.095]  # Example: Move to (1.0, 0.0, 1.0)
            translation_field.setSFVec3f(new_translation)

            # Set new rotation (axis-x, axis-y, axis-z, angle in radians)
            new_rotation = [0, 0, 1, 0]  # Rotate around the Y-axis
            
            rotation_field.setSFRotation(new_rotation)
            self.has_run = True
            self.robot.simulationResetPhysics()
            return Status.SUCCESS
        