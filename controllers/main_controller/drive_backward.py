from py_trees.behaviour import Behaviour
import numpy as np
from py_trees.common import Status

class DriveBackward(Behaviour):
    def __init__(self, name, blackboard, distance):
        """
        Initialize the DriveBackward behavior.

        :param name: Name of the behavior.
        :param blackboard: Shared blackboard for robot data.
        :param distance: Distance to drive backward in meters (positive value).
        """
        super(DriveBackward, self).__init__(name)
        self.robot = blackboard.get('robot')
        self.blackboard = blackboard
        self.distance = distance
        self.start_position = None
        self.target_position = None

    def setup(self):
        self.timestep = self.blackboard.get('timestep')
        self.max_speed = 3.14
        self.leftmotor = self.robot.getDevice('wheel_left_joint')
        self.rightmotor = self.robot.getDevice('wheel_right_joint')

        self.leftmotor.setPosition(float('inf'))
        self.leftmotor.setVelocity(0)

        self.rightmotor.setPosition(float('inf'))
        self.rightmotor.setVelocity(0)

        self.gps = self.robot.getDevice('gps')
        if self.gps is None:
            raise RuntimeError("GPS device not found!")
        self.gps.enable(self.timestep)

        self.compass = self.robot.getDevice('compass')
        if self.compass is None:
            raise RuntimeError("Compass device not found!")
        self.compass.enable(self.timestep)

        self.start_position = None
        self.target_position = None

    def update(self):
        # Get the current GPS position
        gps_values = self.gps.getValues()
        if gps_values is None or len(gps_values) != 3:
            print("Invalid GPS values!")
            return Status.FAILURE

        current_position = np.array([gps_values[0], gps_values[2]])

        # Initialize the starting position and target position
        if self.start_position is None:
            self.start_position = current_position

            # Get the robot's orientation from the compass
            compass_values = self.compass.getValues()
            if compass_values[0] == 0 and compass_values[2] == 0:
                print("Compass values are zero, unable to compute orientation!")
                return Status.FAILURE

            # Calculate the heading angle from the compass
            heading = np.arctan2(compass_values[0], compass_values[2])

            # Compute the direction vector for backward motion
            direction = np.array([-np.sin(heading), -np.cos(heading)])

            # Compute the target position
            self.target_position = self.start_position + direction * self.distance

            print(f"Start Position: {self.start_position}")
            print(f"Target Position: {self.target_position}")
            print(f"Direction Vector: {direction}")

        # Compute the distance to the target position
        distance_to_target = np.linalg.norm(self.target_position - current_position)
        print(f"Distance to target: {distance_to_target:.2f}")

        # Check if the target position is reached
        if distance_to_target < 0.05:  # Tolerance for stopping
            self.leftmotor.setVelocity(0)
            self.rightmotor.setVelocity(0)
            print("Target position reached. Stopping.")
            return Status.SUCCESS

        # Drive backward
        self.leftmotor.setVelocity(-self.max_speed)
        self.rightmotor.setVelocity(-self.max_speed)

        return Status.RUNNING

