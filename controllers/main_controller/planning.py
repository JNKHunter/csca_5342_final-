from controller import Robot,Supervisor
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from scipy import signal
from matplotlib import pyplot as plt 
import numpy as np
import math
from skimage.draw import line_nd

def draw_pixel(display,x,y,color=0xFF0000):
    display.setColor(color)
    display.drawPixel(x,y)

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_nd, random_shapes

# Check if the path between two points is obstacle-free
def IsPathOpen(map, a, b):
    """
    Check if the line from point a to b on the map is free of  obstaclees using Bresenham's algorithm

    Args:
    - map: 2D array representing the environment (False for free space, True for obstacles).
    - a: Tuple (x, y) representing the starting point.
    - b: Tuple (x, y) representing the ending point.

    Returns:
    - True if the path is obstacle-free, False otherwise.
    """
    # Generate the points along the line
    x, y = line_nd(a, b, endpoint=True)

    # Check each point on the line
    for i in range(len(x)):
        # Ensure points are within map bounds
        if not (0 <= x[i] < map.shape[0] and 0 <= y[i] < map.shape[1]):
            return False  # Out of bounds
        
        # Check for obstacles
        if map[x[i], y[i]]:  # Obstacle detected (True values are obstacles)
            return False

    return True

#Find the closest node in the tree
def find_closest_node(graph, qrand):
    min_dist = float("inf")
    closest_node = None
    for node in graph.keys():
        dist = np.sqrt((node[0] - qrand[0])**2 + (node[1] - qrand[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node

# RRT with controlled growth, collision checking, and integer pixel snapping
def grow_rrt_with_pixel_snapping(map, start, goal, Dq, iterations=2500):
    """
    Grow an RRT tree with controlled growth, straight-line collision checking,
    and snapped nodes to integer pixel coordinates.

    Args:
    - map: 2D binary array (False for free space, True for obstacles).
    - start: Tuple (x, y) for the start node.
    - goal: Tuple (x, y) for the goal node.
    - Dq: Step size for controlled growth.
    - iterations: Maximum number of iterations.

    Returns:
    - G: Graph structure of the RRT.
    - parent: Parent dictionary for path reconstruction.
    """
    # Initialize the graph with the start node
    G = {start: []} 
    parent = {start: None}  # Track parent nodes for path reconstruction

    for _ in range(iterations):
        # Sample a random configuration
        if np.random.random() < 0.1:  # 10% chance to bias toward the goal
            qrand = goal
        else:
            qrand = (np.random.randint(0, map.shape[0]), np.random.randint(0, map.shape[1]))

        # Find the closest node in the tree
        qnear = find_closest_node(G, qrand)

        # Compute the direction and step toward qrand
        theta = np.arctan2(qrand[0] - qnear[0], qrand[1] - qnear[1])
        qnew_float = (qnear[0] + Dq * np.sin(theta), qnear[1] + Dq * np.cos(theta))

        # Snap qnew to the nearest pixel
        qnew = (round(qnew_float[0]), round(qnew_float[1]))

        # Check if qnew is within bounds and if the straight line is obstacle-free
        if 0 <= qnew[0] < map.shape[0] and 0 <= qnew[1] < map.shape[1] and IsPathOpen(map, qnear, qnew):
            # Add the new node to the tree
            G[qnear].append((qnew, np.sqrt((qnew[0] - qnear[0])**2 + (qnew[1] - qnear[1])**2)))
            G[qnew] = []
            parent[qnew] = qnear

            # Check if qnew is close enough to the goal and the path is obstacle-free
            if np.sqrt((qnew[0] - goal[0])**2 + (qnew[1] - goal[1])**2) < Dq and IsPathOpen(map, qnew, goal):
                G[qnew].append((goal, np.sqrt((goal[0] - qnew[0])**2 + (goal[1] - qnew[1])**2)))
                G[goal] = []
                parent[goal] = qnew
                break

    return G, parent

class Planning(Behaviour):
    def __init__(self,name,blackboard={},goal=(0,0)):
        super(Planning, self).__init__(name)
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
        self.G,self.parent = grow_rrt_with_pixel_snapping(self.blackboard.get('cspace'), self.start, self.goal,50, iterations=2500) 
        if(len(self.G) > 0):
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
            pixel_path =  []
            current = self.goal
            while current is not None:
                pixel_path.append(current)
                path.append(map2world(current[0],current[1],self.blackboard.get('cspace').shape[0],self.blackboard.get('cspace').shape[1]))
                current = self.parent[current]
            path.reverse()
            self.blackboard['waypoints'] = path
            pixel_path.reverse()
            self.display.setColor(0xADD8E6)
            for i in range(len(pixel_path) - 1):
                x1, y1 = pixel_path[i]
                x2, y2 = pixel_path[i + 1]
                self.display.drawLine(x1, y1, x2, y2)
            return new_status

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
    -235cm and 165cm are the world coordinates for the top left corner of environment 
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

