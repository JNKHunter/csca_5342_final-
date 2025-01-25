from py_trees.behaviour import Behaviour
import numpy as np
from py_trees.common import Status

class Turn180Degrees(Behaviour):
    def __init__(self, name, blackboard):
        super(Turn180Degrees, self).__init__(name)
        self.robot = blackboard.get('robot')
        self.blackboard = blackboard

    def setup(self):
        self.timestep = self.blackboard.get('timestep')
        self.max_speed = 6.28
        self.leftmotor = self.robot.getDevice('wheel_left_joint')
        self.rightmotor = self.robot.getDevice('wheel_right_joint')

        self.leftmotor.setPosition(float('inf'))
        self.leftmotor.setVelocity(0)

        self.rightmotor.setPosition(float('inf'))
        self.rightmotor.setVelocity(0)

        self.compass = self.robot.getDevice('compass')
        if self.compass is None:
            raise RuntimeError("Compass device not found!")
        self.compass.enable(self.timestep)
        '''
        compass_values = self.compass.getValues()
        if compass_values is None or len(compass_values) != 3:
            raise RuntimeError("Invalid compass values!")
        self.initial_heading = np.arctan2(compass_values[0], compass_values[1])

        # Calculate the target heading (180-degree turn)
        self.target_heading = (self.initial_heading + np.pi) % (2 * np.pi)
        self.tolerance = 0.05  # Tolerance for alignment
        '''
        self.has_run = False
    def update(self):
        if not self.has_run:
            compass_values = self.compass.getValues()
            if compass_values is None or len(compass_values) != 3:
                raise RuntimeError("Invalid compass values!")
            self.initial_heading = np.arctan2(compass_values[0], compass_values[1])

            # Calculate the target heading (180-degree turn)
            self.target_heading = (self.initial_heading + np.pi) % (2 * np.pi)
            self.tolerance = 0.05  # Tolerance for alignment
            self.has_run = True

        # Get the current heading
        compass_values = self.compass.getValues()
        print(f"Compass values: {compass_values}")  # Debugging compass values

        if compass_values[0] == 0 and compass_values[1] == 0:
            print("Compass values are zero, unable to compute heading!")
            return Status.FAILURE

        current_heading = np.arctan2(compass_values[0], compass_values[1])
        print(f"Current heading: {current_heading}")  # Debugging current heading

        # Calculate the angular difference to the target
        angle_difference = self.target_heading - current_heading
        print(f'Angle difference 1 = {angle_difference}')
        angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))  # Normalize to [-π, π]
        print(f"Angle difference: {angle_difference}")  # Debugging angular difference

        # Stop motors if within tolerance
        if abs(angle_difference) < self.tolerance:
            self.leftmotor.setVelocity(0)
            self.rightmotor.setVelocity(0)
            print("Alignment achieved. Stopping.")
            return Status.SUCCESS

        # Rotate the robot in place
        rotation_speed = angle_difference * 2.0  # Proportional control, adjust gain as needed
        print(f"Rotation speed: {rotation_speed}")  # Debugging rotation speed
        self.leftmotor.setVelocity(-rotation_speed)
        self.rightmotor.setVelocity(rotation_speed)

        return Status.RUNNING
