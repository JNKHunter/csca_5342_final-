from controller import Robot,Supervisor
from py_trees.behaviour import Behaviour
from py_trees.common import Status
import numpy as np
import math
from collections import deque

def draw_pixel(display,x,y,color=0xFF0000):
    display.setColor(color)
    display.drawPixel(x,y)

import numpy as np

def bfs(grid_map, start, goal):
    queue = deque([start])  # Queue for BFS
    visited = set([start])  # Track visited nodes (initialize with start)
    predecessors = {start: None}  # Store predecessors

    while queue:
        current = queue.popleft()

        # Stop if we reach the goal
        if current == goal:
            print("Goal reached!")
            break

        for neighbor in getNeighbors(current,grid_map):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                predecessors[neighbor] = current

    # Trace back the path
    path = []
    if goal in predecessors:  # If goal is reachable
        node = goal
        while node:
            path.append(node)
            node = predecessors[node]
        path.reverse()

    return path

class PlanningBFS(Behaviour):
    def __init__(self,name,blackboard={},goal=(0,0)):
        super(PlanningBFS, self).__init__(name)
        self.robot = blackboard.get('robot')
        self.cspace = blackboard.get('cspace')
        self.blackboard = blackboard
        self.goal = world2map(goal[0],goal[1],blackboard.get('map_width'),blackboard.get('map_height'))
    def setup(self):
        self.logger.debug(f"Planning::setup {self.name}")
        self.timestep = self.blackboard.get('timestep') 
        self.display = self.robot.getDevice('display')
        robot_start = self.robot.getSelf().getField('translation').getSFVec3f()
        robot_coords = world2map(robot_start[0], robot_start[1],self.blackboard.get('map_width'),self.blackboard.get('map_height')) 
        self.start = robot_coords  
        
    def update(self):
        self.logger.debug(f"Planning::update {self.name}")
        if not self.blackboard.get('is_map_drawn'):
            #Draw the convolution map once. If the map is drawn already, skip the code below
            self.cspace = self.blackboard.get('cspace')
            for row in range(self.cspace.shape[0]):
                for col in range(self.cspace.shape[1]):
                    if self.cspace[row,col]:
                        draw_pixel(self.display,row,col,color=0xFFFFFF)
                    else:
                        draw_pixel(self.display,row,col,color=0x000000) 
            self.blackboard['is_map_drawn'] = True
        #Get the start and goal nodes, and use as input to the grow_rrt function. Return Behavior success if the graph has at least 1 node
        robot_start = self.robot.getSelf().getField('translation').getSFVec3f()
        robot_coords = world2map(robot_start[0], robot_start[1],self.blackboard.get('map_width'),self.blackboard.get('map_height')) 
        self.start = robot_coords  
        self.path = bfs(self.blackboard.get('cspace'), self.start, self.goal) 
        if len(self.path) > 0:
            return Status.SUCCESS
        else:
            return Status.FAILED

    '''
    When this behavior is complete, prepare the shortest path found and save to the blackboard.
    This shortest path will be used by the next navigation behavior.
    '''
    def terminate(self,new_status):
            self.logger.debug(f'Planning::terminate {self.name}')
            # Reconstruct the path from the parent dictionary
            path = []
            pixel_path =  self.path
            print(pixel_path)
            for p in pixel_path:
                path.append(map2world(p[0],p[1],self.blackboard.get('cspace').shape[0],self.blackboard.get('cspace').shape[1]))           
            self.blackboard['waypoints'] = path
            
            self.display.setColor(0xADD8E6)
            for i in range(len(pixel_path) - 1):
                x1, y1 = pixel_path[i]
                x2, y2 = pixel_path[i + 1]
                self.display.drawLine(x1, y1, x2, y2)
            return new_status

# Neighborhood function: Returns neighbors of a cell
def getNeighbors(pos, grid_map):
    # 8 directions: Right, Left, Up, Down, and the 4 diagonals
    directions = [
        (0, 1),  # Right
        (0, -1), # Left
        (-1, 0), # Up
        (1, 0),  # Down
        (-1, 1), # Up-Right
        (-1, -1),# Up-Left
        (1, 1),  # Down-Right
        (1, -1)  # Down-Left
    ]
    neighbors = []
    for di, dj in directions:
        ni, nj = pos[0] + di, pos[1] + dj
        # Check bounds and navigability
        if 0 <= ni < grid_map.shape[0] and 0 <= nj < grid_map.shape[1] and not grid_map[ni, nj]:
            neighbors.append((ni, nj))
    return neighbors

#Function to convert map locations in pixels to world locations in meters 
def map2world(xw,yw,map_width,map_height):
    ''' 
    -235cm and 155cm are the world coordinates for the top left corner of environment 
    I am subtracting/adding these values to orient the top left corner of the arena
    with the top left corner of the map. I'll add a bit of x,y buffer to create a 
    small border around the map as well 
    '''
    px = (xw-235) * 0.01
    py = (155-yw) * 0.01
    #If coords are outside of the bounds, set them to 0
    #print(px,px)
    #if px > map_width - 1 or py > map_height - 1:
    #    return [0,0]
    #else:
    return (px, py) 

#Function to convert world locations in meters to map locations in pixels
def world2map(xw,yw,map_width,map_height):
    ''' 
    -235cm and 155cm are the world coordinates for the top left corner of environment 
    This function is the inverse of map2world above
    '''
    px = math.floor(abs(xw*100 + 235))
    py = math.floor(abs(yw*100 - 155))
    #If coords are outside of the bounds, set them to 0
    #print(px,px)
    if px > map_width - 1 or py > map_height - 1:
        return (0,0)
    else:
        return (px, py)

