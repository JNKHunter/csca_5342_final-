import py_trees
import numpy as np

class Approach(py_trees.behaviour.Behaviour):
    def __init__(self, name, blackboard):
        super(Approach, self).__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.get("robot")
        self.camera = blackboard.get("camera")

    def setup(self):
        self.timestep = int(self.robot.getBasicTimeStep())

        # Initialize wheels
        self.leftMotor = self.robot.getDevice('wheel_left_joint')
        self.rightMotor = self.robot.getDevice('wheel_right_joint')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

        # Define initial motor velocities
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        # Define Center and Length robot needs to be to have a jar based on relative position from camera
        self.reach = 0.85
        self.center = 0.02

    def initialise(self):
        print("Acquiring target")
        self.logger.debug("  %s [Mapping::initialise()]" % self.name)
    
    def update(self):
        error_threshold = 0.01

        # Get camera data
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        objects = self.camera.getRecognitionObjects()

        if len(objects) < 1:
            self.leftMotor.setVelocity(0.5)
            self.rightMotor.setVelocity(-0.5)
            return py_trees.common.Status.RUNNING

        # Find the object that tiago is closest too.
        closest_y = float("inf")
        for i, object in enumerate(objects):
            obj=list(object.getPosition())
            if abs(obj[1]) < closest_y:
                closest_y = abs(obj[1])
                target_obj = obj
        
        # If the object is outside the certain threshold, turn toward the jar 
        if target_obj[1] < -0.03:
            self.leftMotor.setVelocity(0.1)
            self.rightMotor.setVelocity(-0.1)
        else:
            self.leftMotor.setVelocity(-0.1)
            self.rightMotor.setVelocity(0.1)
        
        # If the robot is aligned with the jar, but not close enough to grab it, move forward
        if (-0.03 < target_obj[1] < -0.015) and abs(target_obj[0])-self.reach > error_threshold:
            self.leftMotor.setVelocity(0.2)
            self.rightMotor.setVelocity(0.2)

        # If either alignment or distance to target requirements are not met, then continuing running, otherwise, return success
        if  not (-0.03 < target_obj[1] < -0.015) or target_obj[0]-self.reach > error_threshold:
            return py_trees.common.Status.RUNNING
        else:
            self.leftMotor.setVelocity(0)
            self.rightMotor.setVelocity(0)
            return py_trees.common.Status.SUCCESS
    
    def terminate(self, new_status):
        return super().terminate(new_status)