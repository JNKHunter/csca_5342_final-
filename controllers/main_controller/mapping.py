from os.path import exists
from controller import Robot,Supervisor
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
import math
import os

def draw_pixel(display,x,y,color=0xFF0000):
    display.setColor(color)
    display.drawPixel(x,y)

def world2map(xw,yw,map_width,map_height):
    ''' 
    -215cm and 165cm are the world coordinates for the top left corner of environment 
    I am subtracting these values to orient the top left corner of the arena
    with the top left corner of the map. I'll add a bit of x,y buffer to create a 
    small border around the map as well 
    '''
    px = math.floor(abs(xw*100 + 235))
    py = math.floor(abs(yw*100 - 155))
    #If coords are outside of the bounds, set them to 0
    #print(px,px)
    if px > map_width - 1 or py > map_height - 1:
        return [0,0]
    else:
        return [px, py] 

'''
Simple mapping behavior that generates a probability map with each tick, and when terminated, creates and saves a convolution map.
'''
class Mapping(Behaviour):
    def __init__(self,name,blackboard={}):
        super(Mapping, self).__init__(name)
        self.robot = blackboard.get('robot')
        self.filepath = blackboard.get('filepath')
        self.blackboard = blackboard
        
        self.gps = blackboard.get('gps')
        self.compass = blackboard.get('compass')
        self.lidar = blackboard.get('lidar')
        self.timestep = self.blackboard.get('timestep')
        #Compensate for lidar not oriented at 0,0 in robot coordinates
        self.lidar_translation_x = 0.202
        self.lidar_translation_y = -0.004

        #The Tiago's lidar is inside a box that shields the first and last 80 readings
        self.lidar_drop = 80

        self.display = self.robot.getDevice('display')

        self.map_width = 430
        self.map_height = 567

        self.hasrun = False

        #self.angles = np.linspace(2.0944, -2.0944, len(self.lidar.getRangeImage()))[self.lidar_drop:-self.lidar_drop]
        self.prob_map = np.zeros((self.map_width,self.map_height))



    def update(self):
        self.hasrun = True
        self.logger.debug(f"Mapping::update {self.name}")
        angles = np.linspace(2.0944, -2.0944, len(self.lidar.getRangeImage()))[self.lidar_drop:-self.lidar_drop]
        #Get world coords
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
 
        theta = np.arctan2(self.compass.getValues()[0],self.compass.getValues()[1])

        #Draw robot path
        px,py = world2map(xw,yw,self.map_width,self.map_height)
        draw_pixel(self.display,px,py,color=0xFF0000)

        #The Tiago's Lidar is inside a box that shields the first and last 80 readings.
        ranges = np.array(self.lidar.getRangeImage()[self.lidar_drop:-self.lidar_drop])
        ranges[ranges == np.inf] = 1000

        #Transform robot to world coordinates and calculate the distances to objects
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), xw + self.lidar_translation_x * np.cos(theta) - self.lidar_translation_y * np.sin(theta)],
                     [np.sin(theta), np.cos(theta), yw + self.lidar_translation_x * np.sin(theta) + self.lidar_translation_y * np.cos(theta)],[0, 0, 1]])
    
        X_i = np.array([ranges*np.cos(angles), ranges*np.sin(angles), np.ones((len(ranges),))])
        D = w_T_r @ X_i 
    
        #Convert world coordinates to map coordinates
        map_coordinates = [world2map(xw, yw,self.map_width,self.map_height) for xw, yw in zip(D[0, :], D[1, :])]

        #Draw the map
        self.draw_map(map_coordinates)
        return Status.RUNNING
        
    def draw_map(self,map_coordinates):
        for coord in map_coordinates:
            if self.prob_map[coord[0]][coord[1]] < 1:
                self.prob_map[coord[0]][coord[1]] += 0.01

            v = int(self.prob_map[coord[0]][coord[1]]*255)
            color = v*256**2+v*256+v
            draw_pixel(self.display, coord[0],coord[1], color)

    def terminate(self,new_status):
        self.logger.debug(f"Mapping::terminate {self.name}")
        if self.hasrun:
            kernel = np.ones((55,55))
            cmap = signal.convolve2d(self.prob_map,kernel,mode='same')
            cspace = cmap>0.9
            self.blackboard['cspace'] = cspace
        return new_status
    
class DoesMapExist(Behaviour):
    def __init__(self,name,blackboard):
        super(DoesMapExist, self).__init__(name)
        self.blackboard = blackboard

    def update(self):
        self.logger.debug(f"DoesMapExist::update {self.name}")
        file_exists = exists(self.blackboard.get('filepath'))
        if(file_exists):
            print("Map already exists")
            cspace = np.load(self.blackboard.get('filepath'))
            self.blackboard['cspace'] = cspace
            return Status.SUCCESS
        else:
            print("Map does not exist")
            return Status.FAILURE  