from py_trees.behaviour import Behaviour
from py_trees import logging as log_tree
from py_trees.common import Status

class DetectJamJar(Behaviour):
    def __init__(self, name, blackboard):
        super(DetectJamJar, self).__init__(name=f'DetectJamJar:{name}')
        self.camera = blackboard.get('camera')
        self.blackboard = blackboard

    def update(self):
        objects = self.camera.getRecognitionObjects()
        for obj in objects:
            if 9103 == obj.getId(): 
                self.logger.debug(f"DetectJamJar::update {self.name} found jam jar")
                jam_jar_position = obj.getPosition()
                joint_targets = {
                    'arm_1_joint': jam_jar_position[0], 
                    'arm_2_joint': jam_jar_position[1],
                }
                self.blackboard['joint_targets'] = joint_targets
                print(joint_targets)
                return Status.SUCCESS
        return Status.FAILURE