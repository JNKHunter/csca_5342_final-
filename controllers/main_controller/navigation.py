from controller import Robot,Supervisor
from py_trees.behaviour import Behaviour
from py_trees.common import Status
import numpy as np

#Simple navigation class. Uses the waypoints saved in the blackboard and navigates from point to point until the goal is reached.
class Navigation(Behaviour):
    def __init__(self,name,blackboard={}):
        super(Navigation, self).__init__(name)
        self.robot = blackboard.get('robot')
        self.waypoints = blackboard.get('waypoints')
        self.blackboard = blackboard

    def setup(self):
        self.timestep = self.blackboard.get('timestep')
        self.max_speed = 6.28
        self.leftspeed = self.max_speed
        self.rightspeed = self.max_speed
        self.logger.debug(f"Navigation::setup {self.name}")

        self.leftmotor = self.robot.getDevice('wheel_left_joint')
        self.rightmotor = self.robot.getDevice('wheel_right_joint')

        self.leftmotor.setPosition(float('inf'))
        self.leftmotor.setVelocity(0)

        self.rightmotor.setPosition(float('inf'))
        self.rightmotor.setVelocity(0)

        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep) 
        
        robot_node = self.robot.getSelf() 
        rotation_field = robot_node.getField("rotation")

        self.index = 0 
        self.has_run = False 
        #Mark the waypoint with a physical object
        self.marker = self.robot.getFromDef('marker').getField('translation')

    def update(self):
        print(f'Waypoint:{self.index} of {len(self.waypoints)}')
        print('---- The robot moves VERY slowly for certain waypoints and will take a minute (like waypoint 30 on move to sink) Thank you for your patience!! ----')
        print('---- You can toggle the hide/show render button to speed up navigation!! ----')
        if self.index == len(self.waypoints):
            self.leftspeed = 0
            self.rightspeed = 0
            self.leftmotor.setVelocity(self.leftspeed)
            self.rightmotor.setVelocity(self.rightspeed)
            return Status.SUCCESS

        if not self.has_run:
            self.robot.simulationResetPhysics()
            self.has_run = True

        self.logger.debug(f"Navigation::update {self.name}")
        self.waypoints = self.blackboard.get('waypoints')
        self.marker.setSFVec3f([*self.waypoints[self.index],0])
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]

        #Get the robot's angle and distance from the current waypoint
        theta =np.arctan2(self.compass.getValues()[0],self.compass.getValues()[1])
        rho = np.sqrt((xw-self.waypoints[self.index][0])**2 + (yw-self.waypoints[self.index][1])**2)
        heading = np.arctan2(self.waypoints[self.index][1]-yw, self.waypoints[self.index][0]-xw)
        alpha = heading - theta

        #Make sure alpha stays within +- pi
        if(alpha > np.pi):
            alpha = alpha-2*np.pi

        #Advance the waypoint index if we've reached our target 
        if(rho < 0.3 and self.index < len(self.waypoints)):
            self.index+=1

        #Angle and distance multipliers to speed up the robot
        p1 = 7
        p2 = 4.00
        #Alternate values that seem to have an effect on the stability of the simulation
        #p1 = 4
        #p2 = 2

        self.leftspeed = -alpha*p1 + rho*p2
        self.rightspeed = alpha*p2 + rho*p2

        self.leftspeed = max(min(self.leftspeed,self.max_speed),-self.max_speed)
        self.rightspeed = max(min(self.rightspeed,self.max_speed),-self.max_speed)

        self.leftmotor.setVelocity(self.leftspeed)
        self.rightmotor.setVelocity(self.rightspeed)

        return Status.RUNNING

