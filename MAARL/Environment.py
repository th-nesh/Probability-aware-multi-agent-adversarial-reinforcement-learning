import os
import pickle
import sys
import time
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PRM import apply_PRM, apply_PRM_init, draw_traj, calc_adv_traj, Node, get_traj_edges, get_node_with_coordinates, calc_nearest_dist
from object_detection import Obstacle, apply_object_detection
import copy
from copy import deepcopy
sys.setrecursionlimit(2000)
from time import sleep

ACTION_SPACE_STEP_ADVERSARY = 5
N_ACTIONS_ADVERSARY = 5


class StateAdv1:
    """
    state of the sim-gap-adv
    -- small notes to the states --: basically both agents see the same state (which is "obs") but the other attributes
                                      differ (some stuff to make things work implementation wise...)
                                      also: in general I wanted to distinguish between "observation" and "state" but in
                                      the end didn't do so, so the notation will be kinda random throughout this code
    attributes:
    obs - obs is basically what the state is (the normalized reference map with trajectory)
          in general I wanted to distinguish between observation and state but in the end didn't so the notation will be
          kinda random throughout this code
    traj_index - just an attribut to keep track at which waypoint of the trajectory we are in our episode
                -> you could say traj_index (or rather trajectory[traj_index]) gives us the step, while trajectory is the episode
                but this should not be important anymore since it probably doesn't have to be changed anymore
    position - current position
    angle - current angle
    """
    def __init__(self, obs, pos, angle=0):
        self.obs = obs
        self.traj_index = 1
        self.position = pos
        self.angle = angle

    def __str__(self):
        print('current #traj_node:', self.traj_index, '\n', 'current position:', self.position)


class StateProt:
    """
    -- small notes to the states --: basically both agents see the same state (which is "obs") but the other attributes
                                      differ (some stuff to make things work implementation wise...)
                                      also: in general I wanted to distinguish between "observation" and "state" but in
                                      the end didn't do so, so the notation will be kinda random throughout this code
    state of the protagonist
    attributes:
    obs - obs is basically what the state is (the normalized reference map with trajectory)
    position - current position
    vanilla_cost = total cost of the vanilla trajectory before applying the adversary
    """
    def __init__(self, obs, pos, vanilla_cost):
        self.obs = obs
        self.position = pos
        self.vanilla_cost = vanilla_cost


class Command:
    """
    a command contains the necessary angle, speed and time to move from one waypoint to another
    this is only necessary because we need to model the 'real' process and the influence of the adversary
    """
    def __init__(self, angle, v, t):
        self.angle = angle
        self.v = v
        self.t = t

    def __str__(self):
        return 'rotate: '+str(self.angle)+' speed:'+str(self.v)+' duration:'+str(self.t)

    __repr__ = __str__    # kind of a bad practice


def calc_command(position, current_angle, target):
    """
    calculate the necessary command to (theretically) move from "position" to "target" considering the current angle
    @param position: current position coordinates (pixel location in image)
    @param current_angle:
    @param target: desired position coordinates (pixel location in image)
    @return: Command - object with the necessary rotation angle, speed and time to reach the desired position
    """
    if (position == target).all():
        print('pos', position)      # 101, 103
        print('target', target)
    distance = np.linalg.norm(target - position)
    v = 1   # the factor f = (distance/cost) can be used to scale the intended speed
    # comm_angle is the rotation angle -> positive angles represent counter clockwise rotation
    beta = np.arccos(np.abs(target[0] - position[0]) / distance)
    if current_angle > np.pi:
        current_angle = current_angle - 2*np.pi
    if target[0] - position[0] >= 0 and target[1] - position[1] >= 0:   # 4. Quadrant
        comm_angle = 2 * np.pi - (beta + current_angle)
    elif target[0] - position[0] >= 0 and target[1] - position[1] < 0:  # 1. Quadrant
        comm_angle = beta - current_angle
    elif target[0] - position[0] < 0 and target[1] - position[1] >= 0:  # 3. Quadrant
        comm_angle = np.pi + beta - current_angle
    else:                                                               # 2. Quadrant
        comm_angle = np.pi - (beta + current_angle)

    if comm_angle > np.pi:
        comm_angle = comm_angle - 2*np.pi

    t = distance/v

    command = Command(comm_angle, v, t)

    return command


def calc_new_position(position, current_angle, command):
    """
    function to calculate the new position (basically this function "applies" the command in the simulation)
    @param position: current position
    @param current_angle:
    @param command:
    @return: returns the new position as coordinates in the image as well as the new angle
    """
    x_offset = command.v*command.t*np.cos(command.angle+current_angle)
    y_offset = command.v*command.t*np.sin(command.angle+current_angle)
    new_position_x = np.round(position[0] + x_offset)
    new_position_y = np.round(position[1] - y_offset)
    new_angle = command.angle+current_angle

    return np.array([new_position_x.item(), new_position_y.item()], dtype=int), new_angle


def generate_obs_from_rgb_img(img_grey):
    """
    generates an observation (state) for the agent by normalizing the passed image
    @param img_grey: reference map with original trajectory (might be vanilla, might be the one after prot...)
    @return: normalized version of passed image (observation/state of agent)
    """
    x_max = img_grey.size[0]
    y_max = img_grey.size[1]
    pixels = np.array(img_grey.getdata(), dtype=np.float32).reshape((y_max, x_max))

    v1 = 255
    v2 = 40
    v3 = 100
    v4 = 205
    v5 = 210
    # new_map = Image.fromarray(pixels.astype('uint8'), mode='L')
    # new_map.show()

    pixels[pixels == v1] = 0.6
    pixels[pixels == v2] = 0.2
    pixels[pixels == v3] = 0.4
    pixels[pixels == v4] = 1.
    pixels[pixels == v5] = 0.8

    # uncomment to see what the agent sees (which should be basically a black image)
    # new_map = Image.fromarray(pixels.astype('uint8'), mode='L')
    # new_map.show()

    return pixels


def clear_image_files():
    if os.path.isfile('./image/adv_trajectory.png'):
        try:
            os.remove('./image/adv_trajectory.png')
        except PermissionError:
            print('permissionerror while removing ./image/adv_trajectory.png')


def initialize_map(map_path):
    """
    initializes the reference map by applying the object detection
    @param map_path: path for the scanner map
    @return: reference map and the detected obstacles (with its borders)
    """
    map_ref, obstacles = apply_object_detection(map_path)
    # adding the box that the detection does not see
    order = [0,1,3,2]
    order = [3,0,1,2]
    obstacles[0].corners = [obstacles[0].corners[i] for i in order]
    add = [(1,1),(1,-1),(-1,-1),(-1,1)]
    corners_hand = [(102,123),(103,117),(94,115),(92,120)]
    corners_hand = [tuple(map(lambda i,j:i+j,corners_hand[i],add[i])) for i in range(4)]
    obst_hand = Obstacle(corners_hand)
    obstacles.append(obst_hand)
    # --- only for test we use the already processed map ---
    # obstacles = None
    # ref_map_path = 'map_ref.png'
    # map_ref = Image.open(ref_map_path)
    # obstacle_adversary(obstacles,2)
  

    return map_ref, obstacles

def convert_grid_to_pix(x_r,y_r):
    """
    change from grid cords into pixel cords so that we only move the objects sideways relative to their orientation
    @param cords of the movement in grid cords
    @return tuple of the movement in pixel cords
    """
    x_g = x_r*np.cos(-1.204)-y_r*np.sin(-1.204)
    y_g = x_r*np.sin(-1.204)+y_r*np.cos(-1.204)
    return (x_g,y_g)

def obstacle_adversary(obstacles,action_index):
    """
    initializes the reference map by applying the object detection
    @param obstacles: obstacles in the map
    @param action_index: action of the adversary
    @return: modified obstacles
    """
    action_index = convert_grid_to_pix(*action_index)
    obstacles_disturbed =  obstacles
    #print(obstacles_disturbed)
    for o in range(0,len(obstacles)):
        for i in range(0,len(obstacles[o].corners)):
            obstacles_disturbed[o].corners[i] = tuple(map(lambda j,k : j + k, obstacles_disturbed[o].corners[i], action_index))
        for i in range(0,len(obstacles[o].borders)):
            obstacles_disturbed[o].borders[i][0] = tuple(map(lambda j,k : j + k*np.cos(0), obstacles_disturbed[o].borders[i][0], action_index))
            obstacles_disturbed[o].borders[i][1] = tuple(map(lambda j,k : j + k*np.cos(0), obstacles_disturbed[o].borders[i][1], action_index))

        #     obstacles_disturbed[o].corners[i] = tuple(map(lambda j,k : j + k, obstacles_disturbed[o].corners[i], action_index[o]))
        # for i in range(0,len(obstacles[o].borders)):
        #     obstacles_disturbed[o].borders[i][0] = tuple(map(lambda j,k : j + k*np.cos(0), obstacles_disturbed[o].borders[i][0], action_index[o]))
        #     obstacles_disturbed[o].borders[i][1] = tuple(map(lambda j,k : j + k*np.cos(0), obstacles_disturbed[o].borders[i][1], action_index[o]))
    #print(obstacles_disturbed)
    return obstacles_disturbed

def initialize_traj(map_ref, obstacles=None, nodes=None, visualize=False, edges_all=None, env=None):
    """
    initialize a new usually random trajectory for training
    if we pass nodes here then those nodes are used to calculate a trajectory.
    if no nodes are passed then new nodes are generated (takes time)
    @param map_ref: map that we want to use PRM on - usually the reference map
    @param obstacles: known obstacles (list of Obstacles-objects)
    @param nodes: current nodes (for the graph for PRM)
    @param visualize: determines if the new trajectory should be saved as an image (usually for debugging..)
    @param edges_all: edges_all are the current edges of our graph with their respective costs and nodes. They will
                        be calculated new if we calculate new nodes
    @param env: Environment object. Only needed when dynamically generating random maps
    @return: trajectory - new trajectory as list of nodes
             nodes - new or old nodes, depending on param nodes
             edges_all - edges with their new costs (or unchanged costs, depending on whether we passed nodes)
    """

    if not nodes:
        if env:
            # env.map_ref, env.obstacles = create_random_map()
            traj, traj_opt, nodes, edges_all = apply_PRM_init(env.map_ref, env.obstacles)
        else:
            traj, traj_opt, nodes, edges_all = apply_PRM_init(map_ref, obstacles)
        
        # pickle dump ~~~
        #print('dumping nodes...')
        #open_file = open('nodes_presentation', "wb")
        #pickle.dump(nodes, open_file)
        #open_file.close()

    else:
        #import pdb; pdb.set_trace()
        # for specific start / goal location: ------------------
        start_node = get_node_with_coordinates(nodes,(140,123)) #(66, 69))
        goal_node = get_node_with_coordinates(nodes, (76,98))#(116, 102))
        #traj, _, nodes, _ = apply_PRM(map_ref, nodes, visualize=visualize, start_node=start_node, goal_node=goal_node)
        # ------------------------------------------------------
        # TODO make this part of the config to pass the start and end
        traj, _, nodes, _ = apply_PRM(map_ref, nodes, visualize=visualize,start_node= start_node, goal_node = goal_node)
    # print('fresh trajectory:', traj)

    return traj, nodes, edges_all


def initialize_state_adv1(map_ref, traj, relevant_segments, visualize=False):
    """
    initialize the first state/observation for the sim-gap-adv
    @param map_ref: reference map
    @param traj: current original trajectory for the new state
    @param relevant_segments: how many segments should be considered in the state for the sim-gap-adv
    @param visualize: wether the image files for obsesrvations should be generated
    @return: new State for simulation-gap-adversary
    """
    if relevant_segments != 0:
        relevant_max = min(relevant_segments+1, len(traj))
        traj = traj[:relevant_max]

    map_visu = copy.deepcopy(map_ref)
    draw_traj(map_visu, traj, forAgent=True)
    if visualize:
        map_visu.save('./image/new_obs0.png')

    init_state = StateAdv1(generate_obs_from_rgb_img(map_visu), traj[0].coordinates)

    return init_state


def initialize_state_prot(environment, visualize=False):
    """
    initialize the first state/observation for the protagonist.
    @param environment: Environment object
    @param visualize: wether we want to visualize the actions of prot in image files
    @return: new state/obs of protagonist
    """
    t0 = time.perf_counter()
    map_ref = environment.map_ref
    traj = copy.copy(environment.trajectory_vanilla)
    map_visu = copy.deepcopy(map_ref)
    # adversary = environment.adversary

    draw_traj(map_visu, traj, forAgent=True)

    # if adversary:
    #     init_state = StateAdv1(generate_obs_from_rgb_img(map_visu), traj[0].coordinates)
    #     observation = init_state.obs
    #
    #     traj = [traj[0]]
    #     done = False
    #
    #     while not done:
    #         action_index, _, _, _ = adversary.choose_action(observation, test_mode=True)
    #         observation, _, done, _, node_adv = environment.step_adv1(np.deg2rad(5 * action_index - 10))      # 5 and 10 should be somehow config parameter here (depending on action space of adv1)
    #         traj.append(node_adv)
    #
    #     map_visu = copy.deepcopy(map_ref)   # deepcopy doesn't take much time here
    #     visualize_traj(map_visu, traj, forAgent=True)

    # calculate the nearest distance / point to all objects
    nearest_distances, most_critical_points = calc_nearest_dist(traj, environment.obstacles)
    for i in range(0, len(nearest_distances)):  # gives us the nearest distance / point for each segment in a list
        nearest_distances[i] = np.round(nearest_distances[i][0])
        most_critical_points[i] = most_critical_points[i][0]
        if nearest_distances[i] <= 0:
            nearest_distances[i] = 1        # nearest dist can not be less than 1. the resolution when calculating the distance is not pixel perfect anyway
    nearest_distances = np.array(nearest_distances)
    vanilla_cost = 1/np.min(nearest_distances)
    
    # print('actually before prot and before adversary:', environment.trajectory_vanilla)
    # environment.trajectory_vanilla = traj
    init_state = StateProt(generate_obs_from_rgb_img(map_visu), traj[0].coordinates, vanilla_cost)

    # print('nearest distances:')
    # print(nearest_distances)
    # print('most critical points:')
    # print(most_critical_points)
    # print('vanilla_distance_metric:', vanilla_cost)

    t_init_state = time.perf_counter() - t0
    # print('time init_state (protagonist):', t_init_state)

    if visualize:
        map_visu.save('./image/prot_obs.png')

    return init_state


def copy_nodes(edges):
    """
    copies the passed edges and extracts their nodes and also copies and returns those
    the copies are deepcopies
    @param edges: list of current Edge-objects
    @return: copy of all nodes in edges and copy of edges themselves
    """
    t0 = time.perf_counter()
    edges_copy = copy.deepcopy(edges)
    nodes_copy = []
    for edge in edges_copy:
        if not (edge.node1 in nodes_copy):
            nodes_copy.append(edge.node1)
        if not (edge.node2 in nodes_copy):
            nodes_copy.append(edge.node2)
    print('time_copy_nodes:', time.perf_counter()-t0)
    return nodes_copy, edges_copy



class Environment:
    """
    the environment is the same for both agents, but they use a different step-function

    !!-- some important note --!!: we have to be very careful whenever we change any nodes or edges (or attributes of
                            these) because they are referring to each other and changing one also changes the other.
                            (each edge has nodes and each nodes also has edges)
                            AND: nodes_prot and nodes_vanilla as well as edges_all_vanilla and edges_all_prot are not
                            supposed to be "mixed" -> Node-objects in nodes_prot will be referenced in edges_all_prot
                                                        but are not supposed to be referenced in edges_all_vanilla
                                                      same goes for nodes_vanilla which is not allowed contain/reference
                                                        node objects appearing in edges_all_prot
    attributes:
    adversary - trained adversary (consists of 2 models -> actor and critic) - necessary for step_prot()
    map_ref - reference map -> this is a grayscale image (PIL Image object)
    obstacles - list of Obstacles-objects with their detected borders
    observation_space - size of the reference map represents the observation space. -> this attribute is not used...
    trajectory_vanilla - this is the "original" trajectory just after applying PRM without any agents interfering
                            -> this is a list of Node-objects
    nodes_vanilla - the "original" nodes (or basically the graph), used to generate trajectory_vanilla via PRM
                    -> list of Node-objects
    edges_all_vanilla - all "original" edges connecting the nodes_vanilla. -> list of Edge-objects
                         all_edges_vanilla hahve their costs calculated based on nodes_vanilla and map_ref
    nodes_prot - the nodes (or graph..) after changing the map with protagonist -> used to calculate the new trajectory
                 -> list of Node-objects
    edges_all_prot - all edges with their respective costs after changing the map with protagonist
                    -> list of Edge-objects
    relevant_segments - config parameter that determines how many segments of a trajectory are "seen"
                        "0" means the agents will see the whole trajectory (which there is not so much reason to change)
    state_adv1 - initial state of the sim-gap-adv
    edges_vanilla - edges of the vanilla trajectory (list of Edge-objects)
    done_after_collision - config parameter that determines if the episode (for sim-gap-adv) should terminate after
                            collision was detected or not -> there is not really a reason to not choose it as "True"
    visualize - if this is "True" some png files will be created in the process of training/evaluating the agents
                mainly for debugging (the result is not influenced in any way by this, except it might be a bit slower)
    state_prot - initial state of the protagonist
    """
    def __init__(self, map_path, relevant_segments=0, done_after_collision=True, visualize=False, adversary=None):
        clear_image_files()

        # For now only have angle-offset as action
        # self.action_space = gym.spaces.Discrete(5)          # {-10, -5, 0, 5, 10}
        # self.observation_space = gym.spaces.Box(low=np.full((160, 160), 0), high=np.full((160, 160), 1), shape=(160, 160), dtype=int) # 0: empty, 1: object, 2: trajectory segment (planned), 3: current position
        self.adversary = adversary
        self.map_ref, self.obstacles = initialize_map(map_path)
        self.obstacles_org = copy.deepcopy(self.obstacles)
        self.map_ref_adv = copy.deepcopy(self.map_ref)
        self.observation_space = self.map_ref.size
        self.trajectory_vanilla, self.nodes_vanilla, self.edges_all_vanilla = initialize_traj(self.map_ref, self.obstacles, nodes=None)
        #self.nodes_prot, self.edges_all_prot = copy_nodes(self.edges_all_vanilla)
        self.relevant_segments = relevant_segments
        self.state_adv1 = initialize_state_adv1(self.map_ref, self.trajectory_vanilla, relevant_segments=relevant_segments, visualize=True)
        self.edges_vanilla = get_traj_edges(self.trajectory_vanilla)
        self.done_after_collision = done_after_collision
        self.visualize = visualize
        self.trajectory_adv = []
        self.action_list = []
        #self.state_prot = initialize_state_prot(self, visualize=True)

    
    def modify_map(self):
        self.map_ref_adv = deepcopy(self.map_ref)
        self.map_ref_adv = self.map_ref_adv.convert('RGB')
        map_ref_adv_draw = ImageDraw.Draw(self.map_ref_adv)
        add = [(2,2),(2,-2),(-2,-2),(-2,2)]
        for obstacle in self.obstacles_org:
            # cv2.fillConvexPoly(self.map_ref_adv,obstacle.corners, color='black')
            # increase the size of the obstacle by one pixel
            corners = [tuple(map(lambda i,j:i+j,obstacle.corners[i],add[i])) for i in range(4)]
            map_ref_adv_draw.polygon(corners,fill=(0,0,0),outline=(0,0,0))
        add = [(0,0),(0,-0),(-1,-1),(-0,0)]
        for obstacle in self.obstacles:
            # cv2.fillConvexPoly(self.map_ref_adv,obstacle.corners, color='white')
            corners = [tuple(map(lambda i,j:i+j,obstacle.corners[i],add[i])) for i in range(4)]
            map_ref_adv_draw.polygon(obstacle.corners,fill=(255,255,255),outline=(255,255,255))
        self.map_ref_adv = self.map_ref_adv.convert('L')
        # self.map_ref.show()
        # self.map_ref_adv.show()
        # sleep(2)

    # todo: for now action is just simply the angle-offset but later this should actually be a combination of angle_offset and v_offset
    def step_adv1(self, action, probability,env):
        """
        applies the action of the sim-gap-adv in the environment
        @param action: angle offset (action of adversary)
        @return: obs - the next state/observation
                 reward - reward of this step
                 done - wether this was the last step -> if done==1 the episode terminates here
                 info - tells if we had a collision or not
                 adv1_node2 - only used for applying the adversary after protagonist in step_prot(..)
                 -> this is the new "measured" position, resulting of the adversaries action
        """
        #import pdb; pdb.set_trace()
        done = False
        
        command_vanilla = calc_command(self.state_adv1.position, self.state_adv1.angle, self.trajectory_vanilla[self.state_adv1.traj_index].coordinates)
        ##angle_comm_new = command_vanilla.angle + action
        ##probablity_loc = action_prob(action)
        angle_comm_new = command_vanilla.angle
        v_comm_new = command_vanilla.v
        ##command_disturbed = Command(angle_comm_new, v_comm_new, command_vanilla.t)
        #nearest_distances_van, most_critical_points_van = calc_nearest_dist([self.state_adv1.position, self.trajectory_vanilla[self.state_adv1.traj_index].coordinates], self.obstacles)
        
        # t0 = time.perf_counter()
        
        #print(self.obstacles)
        self.action_list.append(action)
        # TODO 
        # make this change based on amount of obstacles
        obstacles_disturbed = obstacle_adversary(self.obstacles,(action,0))
        #obstacles_disturbed = obstacle_adversary(self.obstacles,(action,action))
        
        #print(self.obstacles
        #print(obstacles_disturbed)
        #print(action)
        pos_new, angle_new = calc_new_position(self.state_adv1.position, self.state_adv1.angle, command_vanilla)
        #trajectory_adversary,_, _ = initialize_traj(self.map_ref, obstacles = obstacles_disturbed, nodes = self.nodes_vanilla, env =env)
        #print(trajectory_adversary)
        #print(self.trajectory_vanilla)
        # this block deals with the situation when the adversary coincidentally steers the robot on the position of the next node in the trajectory (which would end up in a segment with distance 0)
        if not (self.state_adv1.traj_index + 1 >= len(self.trajectory_vanilla)):
            if (pos_new == self.trajectory_vanilla[self.state_adv1.traj_index + 1].coordinates).all():
                self.state_adv1.traj_index += 1
                if self.state_adv1.traj_index >= len(self.trajectory_vanilla):
                    # traj finished (skipping last node because we are already there
                    done = True
        # calc_new_pos_time = time.perf_counter()-t0

        traj_vanilla_coordinates = []
        relevant_max = len(self.trajectory_vanilla)
        if self.relevant_segments != 0:
            relevant_max = min(((self.state_adv1.traj_index + 1) + self.relevant_segments - 1), len(self.trajectory_vanilla))
            for node in self.trajectory_vanilla[self.state_adv1.traj_index + 1:relevant_max]:
                traj_vanilla_coordinates.append(node.coordinates)
        
        segment_adv_coordinates = [self.state_adv1.position, pos_new]
        segment_adv_coordinates.extend(traj_vanilla_coordinates)
        # t3 = time.perf_counter()

        self.modify_map()
        segment_adv_nodes, segments_adv = calc_adv_traj(self.map_ref_adv, segment_adv_coordinates, obstacles_disturbed)
        
        if not self.trajectory_adv:
            self.trajectory_adv.append(segments_adv[0].node1)        
        self.trajectory_adv.append(segments_adv[-1].node2)

        # calc_adv_traj_time = time.perf_counter()-t3
        cost_adv_segments = 0

        for segment in segments_adv:
            cost_adv_segments += segment.cost

        adv1_node1 = segments_adv[0].node1
        adv1_node2 = segments_adv[0].node2

        if self.visualize:
            if os.path.isfile('./image/adv_trajectory.png'):
                visu_adv_traj_map = Image.open('./image/adv_trajectory.png')
            else:
                visu_adv_traj_map = initialize_map(map)
            visu_adv_traj_map = visu_adv_traj_map.convert('RGB')
            visu_adv_traj_map_draw = ImageDraw.Draw(visu_adv_traj_map)
            visu_adv_traj_map_draw.line([(adv1_node1.coordinates[0], adv1_node1.coordinates[1]), (adv1_node2.coordinates[0], adv1_node2.coordinates[1])], fill=(255, 0, 0))
            visu_adv_traj_map_draw.point([(adv1_node1.coordinates[0], adv1_node1.coordinates[1])], fill=(100, 0, 0))
            visu_adv_traj_map_draw.point([(adv1_node2.coordinates[0], adv1_node2.coordinates[1])], fill=(100, 0, 0))
            try:
                visu_adv_traj_map.save('./image/adv_trajectory.png')
                visu_adv_traj_map.save('./image/adv_trajectory_DEBUG.png')
            except PermissionError:
                print('permissionError when saving file')

        # calculation of difference in costs (according to reference map)
        cost_segments_vanilla = 0
        for i in range(0, len(self.edges_vanilla[self.state_adv1.traj_index - 1:relevant_max])):
            cost_segments_vanilla += self.edges_vanilla[self.state_adv1.traj_index - 1 + i].cost
        cost_difference = cost_adv_segments-cost_segments_vanilla
        
        # calculation of difference in distance to closest object...
        # adversary...
    
        nearest_distances_adv, most_critical_points_adv = calc_nearest_dist([self.trajectory_vanilla[self.state_adv1.traj_index-1], self.trajectory_vanilla[self.state_adv1.traj_index]], obstacles_disturbed)
        #print(nearest_distances_adv)
        #print(most_critical_points_adv)
        for i in range(0, len(nearest_distances_adv)):  # gives us the nearest distance / point for each segment in a linst
            nearest_distances_adv[i] = np.round(nearest_distances_adv[i][0])
            most_critical_points_adv[i] = most_critical_points_adv[i][0]
            if nearest_distances_adv[i] <= 0:
                nearest_distances_adv[i] = 1        # nearest dist can not be less than 1. the resolution when calculating the distance is not pixel perfect anyway
        nearest_distances_adv = np.array(nearest_distances_adv)
        #print(nearest_distances_adv)
        adv1_dist_reward = 1/np.min(nearest_distances_adv)
    
        
        # vanilla...
        nearest_distances_vanilla, most_critical_points_vanilla = calc_nearest_dist([self.trajectory_vanilla[self.state_adv1.traj_index-1], self.trajectory_vanilla[self.state_adv1.traj_index]], self.obstacles)
        for i in range(0, len(nearest_distances_vanilla)):  # gives us the nearest distance / point for each segment in a linst
            nearest_distances_vanilla[i] = np.round(nearest_distances_vanilla[i][0])
            most_critical_points_vanilla[i] = most_critical_points_vanilla[i][0]
            if nearest_distances_vanilla[i] <= 0:
                nearest_distances_vanilla[i] = 1        # nearest dist can not be less than 1. the resolution when calculating the distance is not pixel perfect anyway
        nearest_distances_vanilla = np.array(nearest_distances_vanilla)
        vanilla_dist_reward = 1/np.min(nearest_distances_vanilla)

        distance_reward = 5*(adv1_dist_reward)*probability

        collision = 0
        # finally the reward depending on (costs), distance and collision:
        if cost_difference >= 9999:   # colission occured
            reward = 1  # 1
            collision = 1
            if os.path.isfile('./image/adv_trajectory.png'):
                visu_adv_traj_map = Image.open('./image/adv_trajectory.png')
            else:
                #visu_adv_traj_map = copy.deepcopy(Image.open('./image/map_traj.png'))
                # visu_adv_traj_map = copy.deepcopy(self.map_ref)
                visu_adv_traj_map = copy.deepcopy(self.map_ref_adv)
                
            visu_adv_traj_map = visu_adv_traj_map.convert('RGB')
            visu_adv_traj_map_draw = ImageDraw.Draw(visu_adv_traj_map)
            #visu_adv_traj_map_draw.line([(adv1_node1.coordinates[0], adv1_node1.coordinates[1]), (adv1_node2.coordinates[0], adv1_node2.coordinates[1])], fill=(255, 0, 0))
            for i in range(0,self.state_adv1.traj_index):
                visu_adv_traj_map_draw.line([(self.trajectory_vanilla[i].coordinates[0], self.trajectory_vanilla[i].coordinates[1]), (self.trajectory_vanilla[i+1].coordinates[0], self.trajectory_vanilla[i+1].coordinates[1])], fill=(255, 255, 0))
                visu_adv_traj_map_draw.point([(self.trajectory_vanilla[i].coordinates[0], self.trajectory_vanilla[i].coordinates[1])], fill=(200, 255, 0))
                visu_adv_traj_map_draw.point([(self.trajectory_vanilla[i+1].coordinates[0], self.trajectory_vanilla[i+1].coordinates[1])], fill=(200, 255, 0))
           
            for i in range(0,self.state_adv1.traj_index-1):
                visu_adv_traj_map_draw.line([(self.trajectory_adv[i].coordinates[0], self.trajectory_adv[i].coordinates[1]), (self.trajectory_adv[i+1].coordinates[0], self.trajectory_adv[i+1].coordinates[1])], fill=(0, 0, 255))
                visu_adv_traj_map_draw.point([(self.trajectory_adv[i].coordinates[0], self.trajectory_adv[i].coordinates[1])], fill=(0, 0, 255))
                visu_adv_traj_map_draw.point([(self.trajectory_adv[i+1].coordinates[0], self.trajectory_adv[i+1].coordinates[1])], fill=(0, 0, 255))
            
            '''for i in range(0,2):
                for j in range(0, len(obstacles_disturbed[i].corners)):
                    visu_adv_traj_map_draw.point([(obstacles_disturbed[i].corners[j])], fill=(255, 0, 0))
            for i in range(0,2):
                for j in range(0, len(obstacles_disturbed[i].borders)):
                    
                    visu_adv_traj_map_draw.line([obstacles_disturbed[i].borders[j][0], obstacles_disturbed[i].borders[j][1]], fill=(255, 0, 0))
                               
                #visu_adv_traj_map_draw.line([(self.traj_adversary[i][0], self.traj_adversary[i][1]), (self.traj_adversary[i+1][0], self.traj_adversary[i+1][1])], fill=(255, 255, 0))
                #visu_adv_traj_map_draw.point([(self.traj_adversary[i][0], self.traj_adversary[i][1])], fill=(255, 255, 0))
                #visu_adv_traj_map_draw.point([(self.traj_adversary[i+1][0], self.traj_adversary[i+1][1])], fill=(255, 255, 0))'''
            try:
                visu_adv_traj_map.save('./image/adv_trajectory.png')
                visu_adv_traj_map.save('./image/adv_trajectory_DEBUG.png')
            except PermissionError:
                print('permissionError when saving file')
            if self.done_after_collision:
                done = True
                action_sum=  sum(self.action_list)
                print(self.action_list)
                # self.obstacles = obstacle_adversary(obstacles_disturbed, (-action_sum,0))
                #self.obstacles = obstacle_adversary(obstacles_disturbed, (-action_sum,-action_sum))
                self.action_list = []

                # print('collision')
        else:
            reward = distance_reward
        

        update_map_each_step = False
        if update_map_each_step:
            visu_adv_traj_map = copy.deepcopy(self.map_ref_adv)
                
            visu_adv_traj_map = visu_adv_traj_map.convert('RGB')
            visu_adv_traj_map_draw = ImageDraw.Draw(visu_adv_traj_map)
            #visu_adv_traj_map_draw.line([(adv1_node1.coordinates[0], adv1_node1.coordinates[1]), (adv1_node2.coordinates[0], adv1_node2.coordinates[1])], fill=(255, 0, 0))
            for i in range(0,self.state_adv1.traj_index):
                visu_adv_traj_map_draw.line([(self.trajectory_vanilla[i].coordinates[0], self.trajectory_vanilla[i].coordinates[1]), (self.trajectory_vanilla[i+1].coordinates[0], self.trajectory_vanilla[i+1].coordinates[1])], fill=(255, 255, 0))
                visu_adv_traj_map_draw.point([(self.trajectory_vanilla[i].coordinates[0], self.trajectory_vanilla[i].coordinates[1])], fill=(200, 255, 0))
                visu_adv_traj_map_draw.point([(self.trajectory_vanilla[i+1].coordinates[0], self.trajectory_vanilla[i+1].coordinates[1])], fill=(200, 255, 0))
            
            for i in range(0,self.state_adv1.traj_index-1):
                visu_adv_traj_map_draw.line([(self.trajectory_adv[i].coordinates[0], self.trajectory_adv[i].coordinates[1]), (self.trajectory_adv[i+1].coordinates[0], self.trajectory_adv[i+1].coordinates[1])], fill=(0, 0, 255))
                visu_adv_traj_map_draw.point([(self.trajectory_adv[i].coordinates[0], self.trajectory_adv[i].coordinates[1])], fill=(0, 0, 255))
                visu_adv_traj_map_draw.point([(self.trajectory_adv[i+1].coordinates[0], self.trajectory_adv[i+1].coordinates[1])], fill=(0, 0, 255))
            
            '''for i in range(0,2):
                for j in range(0, len(obstacles_disturbed[i].corners)):
                    visu_adv_traj_map_draw.point([(obstacles_disturbed[i].corners[j])], fill=(255, 0, 0))
            for i in range(0,2):
                for j in range(0, len(obstacles_disturbed[i].borders)):
                    
                    visu_adv_traj_map_draw.line([obstacles_disturbed[i].borders[j][0], obstacles_disturbed[i].borders[j][1]], fill=(255, 0, 0))
                                
                #visu_adv_traj_map_draw.line([(self.traj_adversary[i][0], self.traj_adversary[i][1]), (self.traj_adversary[i+1][0], self.traj_adversary[i+1][1])], fill=(255, 255, 0))
                #visu_adv_traj_map_draw.point([(self.traj_adversary[i][0], self.traj_adversary[i][1])], fill=(255, 255, 0))
                #visu_adv_traj_map_draw.point([(self.traj_adversary[i+1][0], self.traj_adversary[i+1][1])], fill=(255, 255, 0))'''
            try:
                visu_adv_traj_map.save('./image/adv_trajectory.png')
                visu_adv_traj_map.save('./image/adv_trajectory_DEBUG.png')
            except PermissionError:
                print('permissionError when saving file')

        self.state_adv1.position = pos_new
        self.state_adv1.angle = angle_new
        self.state_adv1.traj_index += 1
        relevant_max = min(relevant_max+1, len(self.trajectory_vanilla))

        if self.state_adv1.traj_index >= len(self.trajectory_vanilla):
            done = True
            action_sum=  sum(self.action_list)
            print(self.action_list)
            # self.obstacles = obstacle_adversary(obstacles_disturbed, (-action_sum,0))
            #self.obstacles = obstacle_adversary(obstacles_disturbed, (-action_sum,-action_sum))

            self.action_list = []

        if not done:
            segments_vanilla = [Node(pos_new[0], pos_new[1])]
            segments_vanilla.extend(self.trajectory_vanilla[self.state_adv1.traj_index:relevant_max])
            map_visu_new = copy.deepcopy(self.map_ref)
            # t1 = time.perf_counter()
            draw_traj(map_visu_new, segments_vanilla, forAgent=True)
            # visualize_traj_time = time.perf_counter()-t1
            #if self.visualize:
            #    map_visu_new.save('./image/new_obs' + str(self.state_adv1.traj_index - 1) + '.png')    # debug
            # t4 = time.perf_counter()
            self.state_adv1.obs = generate_obs_from_rgb_img(map_visu_new)
            # generate_obs_time = time.perf_counter()-t4
            
        info = collision
        
        self.obstacles = obstacle_adversary(obstacles_disturbed, (-action,0))

        return self.state_adv1.obs, reward, done, info, adv1_node2

    

    def reset(self, relevant_agent, reset_traj=True, new_nodes=False):
        """
        resets the environment - that means samples a new trajectory and if "new_nodes" is true then also samples new
        nodes (and therefor graph) for PRM
        @param relevant_agent: depending on which agent is trained the reset function has to behave a bit different
        @param reset_traj: telling the reset function if the trajectory should be reset (which will usually be True)
        @param new_nodes: telling wether we want to sample a new graph or keep the current one
        @return: returns obs_ret, which is the new state for the "relevant_agent"
        """
        clear_image_files()

        if reset_traj:
            if not new_nodes:
                self.trajectory_vanilla, self.nodes_vanilla, _ = initialize_traj(self.map_ref, nodes=self.nodes_vanilla, visualize=self.visualize)
            else:
                self.trajectory_vanilla, self.nodes_vanilla, self.edges_all_vanilla = initialize_traj(self.map_ref, obstacles=self.obstacles, visualize=self.visualize, env=self)
                self.nodes_prot, self.edges_all_prot = copy_nodes(self.edges_all_vanilla)
        self.edges_vanilla = get_traj_edges(self.trajectory_vanilla)
        obs_ret = None
        if relevant_agent == 'adv1':
            self.state_adv1 = initialize_state_adv1(self.map_ref, self.trajectory_vanilla, relevant_segments=self.relevant_segments, visualize=self.visualize)
            obs_ret = self.state_adv1.obs
        elif relevant_agent == 'prot':
            self.state_adv1.traj_index = 1
            self.state_prot = initialize_state_prot(self)
            obs_ret = self.state_prot.obs
        self.trajectory_adv = []
        # print('Fresh:', self.trajectory_vanilla)

        return obs_ret
