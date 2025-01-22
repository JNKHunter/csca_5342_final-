from controller import Robot,Supervisor
from py_trees.behaviour import Behaviour
from py_trees.common import Status
import py_trees
import numpy as np

def get_sensor_name_for_joint(joint_name):
    if 'gripper_left_finger_joint' == joint_name:
        sensor_name = 'gripper_left_sensor_finger_joint'
    elif 'gripper_right_finger_joint' == joint_name:
        sensor_name = 'gripper_right_sensor_finger_joint'
    else:
        sensor_name = f"{joint_name}_sensor"
    return sensor_name

class ServoArm(py_trees.behaviour.Behaviour):
    def __init__(self, name, joint_targets, blackboard, threshold=0.001):
        super(ServoArm, self).__init__(name=f'ServoArm:{name}')
        self.robot = blackboard.get('robot')
        self.joint_targets = joint_targets
        print(joint_targets)
        self.timestep = blackboard.get('timestep')
        self.threshold = threshold  # Error threshold to stop movement
        self.motors = {}
        self.encoders = {}

        # Initialize motors and encoders for each joint
        for joint_name, target_position in self.joint_targets.items():
            motor = self.robot.getDevice(joint_name)
            encoder = self.robot.getDevice(get_sensor_name_for_joint(joint_name))
            encoder.enable(blackboard.get('timestep'))
            motor.setPosition(float('inf'))  # Enable velocity control
            #motor.setVelocity(1.0)  # Set a default velocity
            self.motors[joint_name] = motor
            self.encoders[joint_name] = encoder

    def update(self):
        total_squared_error = 0.0

        # Calculate the cumulative squared error for all joints
        for joint_name, target_position in self.joint_targets.items():
            current_position = self.encoders[joint_name].getValue()
            error = target_position - current_position
            total_squared_error += error ** 2

            # Set the motor position to move closer to the target
            self.motors[joint_name].setPosition(target_position)

        # Check if the total error is below the threshold
        if total_squared_error < self.threshold:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

