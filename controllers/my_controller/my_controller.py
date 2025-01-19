"""my_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import py_trees

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
compass = robot.getDevice('compass')
gps = robot.getDevice('gps')
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
leftmotor = robot.getDevice('wheel_left_joint')
rightmotor = robot.getDevice('wheel_right_joint')

# Dictionary with joint names and their initial positions
robot_joints = {
	'torso_lift_joint': 0.35,
	'arm_1_joint': 0.71,
	'arm_2_joint': 1.02,
	'arm_3_joint': -2.815,
	'arm_4_joint': 1.011,
	'arm_5_joint': 0,
	'arm_6_joint': 0,
	'arm_7_joint': 0,
	'gripper_left_finger_joint': 0,
	'gripper_right_finger_joint': 0,
	'head_1_joint': 0,
	'head_2_joint': 0
}

# Dictionary to store encoder handles
encoders = {}
devices = {}

# Loop through the joint names to create the encoder dictionary
for joint_name in robot_joints.keys():
	if 'gripper_left_finger_joint' == joint_name:
		sensor_name = 'gripper_left_sensor_finger_joint'
	elif 'gripper_right_finger_joint' == joint_name:
		sensor_name = 'gripper_right_sensor_finger_joint'
	else:
		sensor_name = f"{joint_name}_sensor"
	
	encoder = robot.getDevice(sensor_name)  # Get the encoder device
	encoder.enable(timestep)  # Enable the encoder with the simulation timestep
	encoders[joint_name] = encoder  # Store the encoder in the dictionary

for joint_name, target_position in robot_joints.items():
	motor = robot.getDevice(joint_name)  # Get the motor device
	motor.setPosition(target_position)  # Set the target position
	devices[joint_name] = motor  # Store the motor handle
has_run = False


import py_trees
from controller import Robot

class ServoJoint(py_trees.behaviour.Behaviour):
	def __init__(self, robot, joint_name, target_position, timestep):
		super(ServoJoint, self).__init__(name=f"ServoJoint_{joint_name}")
		self.robot = robot
		self.joint_name = joint_name
		self.target_position = target_position
		self.timestep = timestep
		self.motor = self.robot.getDevice(joint_name)
		self.encoder = self.robot.getDevice(f"{joint_name}_sensor")
		self.encoder.enable(timestep)
		self.motor.setPosition(float('inf'))  # Enable velocity control
		self.motor.setVelocity(1.0)  # Set a default velocity
		self.reached = False

	def update(self):
		current_position = self.encoder.getValue()
		if abs(current_position - self.target_position) < 0.01:  # Tolerance
			self.reached = True
			return py_trees.common.Status.SUCCESS
		elif not self.reached:
			self.motor.setPosition(self.target_position)
			return py_trees.common.Status.RUNNING
		else:
			return py_trees.common.Status.FAILURE

class ServoArm(py_trees.behaviour.Behaviour):
	def __init__(self, robot, joint_targets, timestep, threshold=0.001):
		super(ServoArm, self).__init__(name="ServoArm")
		self.robot = robot
		self.joint_targets = joint_targets
		self.timestep = timestep
		self.threshold = threshold  # Error threshold to stop movement
		self.motors = {}
		self.encoders = {}

		# Initialize motors and encoders for each joint
		for joint_name, target_position in joint_targets.items():
			motor = self.robot.getDevice(joint_name)
			encoder = self.robot.getDevice(f"{joint_name}_sensor")
			encoder.enable(timestep)
			motor.setPosition(float('inf'))  # Enable velocity control
			motor.setVelocity(1.0)  # Set a default velocity
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

class ServoPartialArm(py_trees.behaviour.Behaviour):
	def __init__(self, robot, joint_targets, timestep):
		super(ServoPartialArm, self).__init__(name="ServoPartialArm")
		self.robot = robot
		self.joint_targets = joint_targets
		self.timestep = timestep
		self.reached = {joint_name: False for joint_name in joint_targets.keys()}

	def update(self):
		all_reached = True
		for joint_name, target_position in self.joint_targets.items():
			motor = self.robot.getDevice(joint_name)
			encoder = self.robot.getDevice(f"{joint_name}_sensor")
			encoder.enable(self.timestep)

			current_position = encoder.getValue()
			if abs(current_position - target_position) > 0.01:  # Tolerance
				motor.setPosition(target_position)
				all_reached = False
			else:
				self.reached[joint_name] = True

		if all_reached:
			return py_trees.common.Status.SUCCESS
		return py_trees.common.Status.RUNNING


# Define the target positions for selected joints
joint_targets = {
	'arm_1_joint': 1.0,
	'arm_3_joint': -1.5,
	'arm_5_joint': 0.5,
}

# Set the error threshold
threshold = 0.1

# Create the ServoArm behavior tree node
servo_arm_node = ServoArm(robot, joint_targets, timestep, threshold)

# Create a root for the PyTrees tree
root = py_trees.composites.Sequence(name="Root", memory=True)
root.add_child(servo_arm_node)

# Run the PyTrees behavior tree
while robot.step(timestep) != -1:
	status = root.tick_once()
	if status == py_trees.common.Status.SUCCESS:
		print("All joints reached their target positions.")
		break
	elif status == py_trees.common.Status.RUNNING:
		print("Moving joints to target positions...")

