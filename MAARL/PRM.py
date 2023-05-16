import math
import time
import numpy as np
from PIL import Image, ImageDraw
import random
import copy
import time
import sys

import pickle

sys.setrecursionlimit(2000)

# todo: these parameters should also be part of the config file...
N_NODES = 300       # amount of nodes sampled in the map
N_NEIGHBOURS = 20   # amount of neighbours for each node -> the "k" of "k-nearest neighbour"
RADIUS_ROBOT = 6    # 5/6 seems to represent the robotino quite good

# don't change these COLOR values (they are used in Environment.py hardcoded..)
NODE_COLOR = 160
EDGE_COLOR = 50
START_COLOR = 205
GOAL_COLOR = 120
TRAJ_COLOR = 100
TRAJ_NODE_COLOR = 210
ADV_TRAJ_COLOR = 35
ADV_TRAJ_NODE_COLOR = 110
COLOR_UNKNOWN_AREA = 40

COST_BLACK_PIXEL = 1
COST_WHITE_PIXEL = 99999


class Node:
    """
    Nodes for PRM - Every node has an arbitrary number of neighbours (other nodes that are conneted via edges)
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = np.array([x, y])
        self.neighbours = []
        self.edges = []
        # for deijkstra
        self.tentative_dist = math.inf
        self.visited = False
        self.predecessor = None

    def __str__(self):
        return str(self.coordinates)

    __repr__ = __str__    # X kind of a bad practice


class Edge:
    """
    Edges for PRM - Every edge has exactly two nodes (which it connects to)
    attributs:
    length - length of the edge -> doesn't regard the "color" of the traversed pixels
    cost - cost value of the edge -> depends on length of the edge and the color of traversed pixels
    edge_points - a list of coordinates that interpolate the edge (every second pixel only for performance reasons)
                -> is used when checking for distance to obstacles
    """
    def __init__(self, node1, node2, length, cost=0):
        self.node1 = node1    # start of edge
        self.node2 = node2    # end of edge
        self.length = length
        self.cost = cost
        self.edge_points = []

    def set_cost(self, cost):
        self.cost = cost

    def __str__(self):
        node1_str = str(self.node1)
        node2_str = str(self.node2)
        return str(node1_str+'<--->'+node2_str+'  length: '+'%.2f' % self.length+' cost: '+'%.2f' % self.cost)

    __repr__ = __str__    # X kind of a bad practice


def add_nodes(map_ref, N, obstacles, start=None, goal=None):
    """
    adds the nodes of the graph into the reference map for PRM
    -> since the robot has a width, we don't place the nodes close to the obstacles (->RADIUS_ROBOT)
    @param map_ref: reference map
    @param N: amount of nodes to sample
    @param obstacles: list of Obstacle-objects (results from object detection)
    @param start: if we want a specific position to be sampled as start we have to pass it here (as tuple)
    @param goal: if we want a specific position to be sampled as goal we have to pass it here (as tuple)
    @return: im_with_nodes - PIL Image of the reference map mit the nodes added as grey dots - just for visualization
             nodes - list of Node-objects
    """
    im_width = map_ref.size[0]
    im_height = map_ref.size[1]

    pixels = np.array(map_ref.getdata()).reshape((im_height, im_width))
    N_nodes = 0

    nodes = []
    if start and goal:
        nodes.append(Node(start[0], start[1]))
        nodes.append(Node(goal[0], goal[1]))
        N_nodes = 2

    while N_nodes < N:
        random_x = np.random.randint(im_width)
        random_y = np.random.randint(im_height)
        distances = []
        for obstacle in obstacles:
            distances.append(obstacle.distance_to_point((random_x, random_y)))
        dist = np.min(np.array(distances))
        if (dist > RADIUS_ROBOT) and pixels[random_y][random_x] != 255 and pixels[random_y][random_x] != NODE_COLOR:
            pixels[random_y][random_x] = NODE_COLOR
            N_nodes += 1
            nodes.append(Node(random_x, random_y))

    im_with_nodes = Image.fromarray(pixels.astype('uint8'), mode='L')
    print(len(nodes))
    return im_with_nodes, nodes


def calculate_edge_costs(map_ref, edges, obstacles=None, prot=False):
    """
    recalculates the cost of all passed edges based on the passed reference map and obstacles
    PROBLEM of calculate_edge_cost: Can only detect collisions with known obstacles
    therefore there is another function calc_cost, see comments there for more information
    @param map_ref: reference map
    @param edges: list of Edge-objects that should be recalculated
    @param obstacles: list of Obstacle-objects
    @param prot: if the function is called to calculate the weights for the protagonist, we have to return (only) the
            edges, which have been influenced by the protagonist
    @return: returns the edges that has been changed by the protagonist (if there was prot=True)
    """
    # t0 = time.perf_counter()
    map_size = map_ref.size
    map_matrix = np.reshape(np.array(map_ref.getdata()), (map_size[0], map_size[1]))
    edges_prot = {}

    for edge in edges:
        if edge.cost >= COST_WHITE_PIXEL:  # if edge cost is causing a collision already (edge cost >= COST_COLLISION) we don't have to recalculate its weight (after prot it can only be even higher)
            continue
        else:
            edge_cost = 0
            distances = []
            close_object = False
            i = 0
            rounded_edge_points = np.round(np.array(edge.edge_points)).astype(int)
            rounded_edge_points_x = rounded_edge_points[:, 0]
            rounded_edge_points_y = rounded_edge_points[:, 1]
            grayscale_vals = map_matrix[rounded_edge_points_y, rounded_edge_points_x]
            for edge_point in edge.edge_points:
                grayscale_val = grayscale_vals[i]
                pixel_cost = grayscale_to_cost(grayscale_val)
                edge_cost += pixel_cost
                # check only for every second point on edge to save some time
                i += 1
                if obstacles:
                    if (i % 2 == 0) and (not close_object) and (edge_cost < COST_WHITE_PIXEL):
                        for obstacle in obstacles:
                            distances.append(obstacle.distance_to_point(edge_point))
                        if np.min(np.array(distances)) < RADIUS_ROBOT:
                            close_object = True
            if close_object:    # if the next object border is closer than the robot radius, driving on the current edge causes a collision -> adjust edge cost
                edge_cost += COST_WHITE_PIXEL

            if prot:
                if edge_cost >= COST_WHITE_PIXEL:
                    edges_prot[edge] = edge.cost
            

            edge.cost = edge_cost
    # print('prot edges time (fill dict):', t_prot_edges)
    # print('time calculate edge costs:', time.perf_counter()-t0)
    return edges_prot


def interpolate_segment(segment):
    """
    a list of coordinates that interpolate the passed semgment (every second pixel only for performance reasons)
                -> is used when checking for distance to obstacles
    @param segment: segment of a trajectory (from one waypoint (Node) to another - this is also just one Edge)
    @return: segment_interpolated is a list of coordinates that interpolates the segment
    """
    p1 = segment[0]
    p2 = segment[1]

    interpol_stepsize = 2
    length = np.linalg.norm(np.array(p2) - np.array(p1))

    n_interpol_steps = int(length/interpol_stepsize)
    # print('n_interpol_steps', n_interpol_steps)

    segment_interpolated = []

    if n_interpol_steps != 0:
        step_x = (p2[0]-p1[0])/n_interpol_steps
        step_y = (p2[1]-p1[1])/n_interpol_steps

        for i in range(0, n_interpol_steps+1):
            segment_interpolated.append((np.round(p1[0]+i*step_x), np.round(p1[1]+i*step_y)))
    else:
        segment_interpolated.append((p1[0], p1[1]))
        segment_interpolated.append((p2[0], p2[1]))

    return segment_interpolated


def calc_nearest_dist(traj, obstacles):
    """
    calculates the shortest distances for every segment of the passed trajectory to all of the known obstacles
    analytically (using the borders of the obstacles)
    @param traj: trajectory of which the shortest distance is supposed to be calculated
    @param obstacles: list of Obstacle-objects
    @return: closest_distances_all - list of distance values (one value for every segment)
             closest_points_all - list of tuples.
                                    every tuple represents location of the point on the corresponding segment that has
                                    the smallest distance to an object
    -- small note --: in the end the algorithm will just make use of the minimum value of the "closest_distances_all"
                        i.e. considering only the nearest (point) to an obstacle
    """
    closest_distances_all = []
    closest_points_all = []
    for i in range(0, len(traj)-1):
        segment = [traj[i].coordinates, traj[i+1].coordinates]

        segment_interpolated = interpolate_segment(segment)

        closest_distances = []
        closest_points = []
        for obstacle in obstacles:
            for interp_point in segment_interpolated:
                dist = obstacle.distance_to_point(interp_point)
                closest_distances.append(dist)
                closest_points.append(np.array(interp_point))

        indexes_closest_dist = np.argsort(np.array(closest_distances))
        closest_distances = np.array(closest_distances)[indexes_closest_dist]
        closest_points = np.array(closest_points)[indexes_closest_dist]
        closest_distances_all.append(closest_distances.tolist())
        closest_points_all.append(closest_points.tolist())

    return closest_distances_all, closest_points_all


def add_neighbours(map_ref, nodes, N):
    """
    applies the k-nearest neighbours algorithm to construct a graph
    @param map_ref: reference map
    @param nodes: nodes to build the graph of
    @param N: "k" of k-nearest neighbour -> amount of other nodes to connect for each node
    @return: map_ref_copy1 - reference map (unchanged)
             nodes - list of nodes, that now have neighbours and edges
             edges_all - list of Edge-objects, all edges in the graph with their cost values
    """
    deepcopy_total_time = 0
    calc_cost_total_time = 0
    draw_line_total_time = 0
    calc_all_distances_total_time = 0

    t0 = time.perf_counter()
    map_ref_copy_1 = copy.deepcopy(map_ref)
    deepcopy_total_time += time.perf_counter()-t0

    map_ref_draw = ImageDraw.Draw(map_ref_copy_1)
    edges_all = []

    t0 = time.perf_counter()
    map_ref_copy_2 = copy.deepcopy(map_ref)
    deepcopy_total_time += time.perf_counter()-t0
    for node_i in range(0, len(nodes)):
        available_nodes = copy.copy(nodes)
        available_nodes.remove(nodes[node_i])
        for neighbour in nodes[node_i].neighbours:
            available_nodes.remove(neighbour)
        for k in range(0, N):
            t0 = time.perf_counter()
            map_visu = copy.deepcopy(map_ref_copy_2)
            deepcopy_total_time += time.perf_counter()-t0

            map_visu_draw = ImageDraw.Draw(map_visu)

            if len(available_nodes) != 0:
                shortest_dist = math.inf
                if len(nodes[node_i].neighbours) >= N:
                    edge_distances = []
                    for edge in nodes[node_i].edges:
                        edge_distances.append(edge.length)
                    shortest_dist = np.max(edge_distances)

                # most time consuming part: --------------------------v
                t3 = time.perf_counter()
                closest_neighbour = None
                for available_node_j in range(0, len(available_nodes)):
                    dist = np.linalg.norm(nodes[node_i].coordinates - available_nodes[available_node_j].coordinates)
                    if dist < shortest_dist:
                        shortest_dist = dist
                        closest_neighbour = available_nodes[available_node_j]
                calc_all_distances_total_time += time.perf_counter()-t3
                # ----------------------------------------------------^

                if closest_neighbour is not None:
                    # calculate length and cost of edge
                    t2 = time.perf_counter()
                    map_ref_draw.line([(nodes[node_i].coordinates[0], nodes[node_i].coordinates[1]), (closest_neighbour.coordinates[0], closest_neighbour.coordinates[1])], fill=EDGE_COLOR)
                    map_visu_draw.line([(nodes[node_i].coordinates[0], nodes[node_i].coordinates[1]), (closest_neighbour.coordinates[0], closest_neighbour.coordinates[1])], fill=EDGE_COLOR)
                    draw_line_total_time += time.perf_counter() - t2
                    t1 = time.perf_counter()
                    cost = calc_cost(map_ref_copy_2, map_visu, nodes[node_i].coordinates)
                    calc_cost_total_time += time.perf_counter()-t1
                    # create new edge
                    edge = Edge(nodes[node_i], closest_neighbour, length=shortest_dist, cost=cost)
                    if edge not in edges_all:
                        edges_all.append(edge)
                    # append neighbour and edge for node_i
                    nodes[node_i].neighbours.append(closest_neighbour)
                    nodes[node_i].edges.append(edge)
                    # append neighbour and edge for closest_neighbour
                    closest_neighbour.neighbours.append(nodes[node_i])
                    closest_neighbour.edges.append(edge)

                    map_ref_draw.point([(nodes[node_i].coordinates[0], nodes[node_i].coordinates[1])], fill=NODE_COLOR)
                    map_ref_draw.point([(closest_neighbour.coordinates[0], closest_neighbour.coordinates[1])], fill=NODE_COLOR)

                    available_nodes.remove(closest_neighbour)

    for edge in edges_all:
        p1 = edge.node1.coordinates
        p2 = edge.node2.coordinates

        length = np.linalg.norm(p2 - p1)

        step_x = (p2[0]-p1[0])/length
        step_y = (p2[1]-p1[1])/length

        edge_points = [p1]
        for i in range(0, int(length)+1):
            point = np.array([np.round(p1[0]+i*step_x), np.round(p1[1]+i*step_y)])
            if not (point == edge_points[-1]).all():
                edge.edge_points.append((np.round(p1[0]+i*step_x), np.round(p1[1]+i*step_y)))

    # print('deepcopy_total_time:', deepcopy_total_time)
    # print('calc_cost_total_time:', calc_cost_total_time)
    # print('draw_line_total_time:', draw_line_total_time)
    # print('calculate_all_distances_total_time:', calc_all_distances_total_time)
    return map_ref_copy_1, nodes, edges_all


def deijkstra(nodes, start, goal):
    """
    -- small note --: I didn't account for "impossible" trajectories where deijkstra can't find the goal
                        -> islands in graph
    this function applies the dijkstra algorithm and returns a list of nodes that represent the optimal trajectory
    @param nodes: list of Node-Objects that need the have neighbours added (->graph) at this point
    @param start: Node - start node of the calculated trajectory
    @param goal: Node - goal node of the calculated trajectory
    @return: "shortest" trajectory based on the passed graph. "shortest" means optimal in terms of the edge costs, not
             necessarily in euclidean length.
    """
    # init
    nodes_copy = copy.copy(nodes)
    nodes_unvisited = nodes
    current_node = start
    start.tentative_dist = 0
    done = False
    trajectory = []

    # loop
    while not done:
        for neighbour in current_node.neighbours:
            if not neighbour.visited:
                pos_new_tentative = find_common_edge(current_node, neighbour).cost + current_node.tentative_dist
                if pos_new_tentative < neighbour.tentative_dist:
                    neighbour.tentative_dist = pos_new_tentative
                    neighbour.predecessor = current_node
        current_node.visited = True
        nodes_unvisited.remove(current_node)

        # check if we reached the goal already and build up optimal trajectory
        if current_node == goal or len(nodes_unvisited) == 0:       # found the shortest path or no path # XX todo: what if we found no path? -> len(nodes_unvisited) == 0
            done = True
            traj_node = goal
            trajectory.append(goal)
            while traj_node.predecessor is not None:
                trajectory.insert(0, traj_node.predecessor)
                traj_node = traj_node.predecessor

        # not finished yet... find the next node we want to visit
        if not len(nodes_unvisited) == 0:
            smallest_tent = math.inf
            smallest_tent_node = None
            for unv_node in nodes_unvisited:
                if unv_node.tentative_dist < smallest_tent:
                    smallest_tent = unv_node.tentative_dist
                    smallest_tent_node = unv_node
            current_node = smallest_tent_node
    for node in nodes_copy:
        node.tentative_dist = math.inf
        node.visited = False
        node.predecessor = None
    return trajectory


# recursive function to "follow" a line and sum up its costs
def calc_cost(ref_map, colored_map, coordinates):
    """
    this function "graphically" calculates the costs of an edge with the floodfill algorithm
    -> this function is still used to calculate the cost of edges. ALTHOUGH there is also the function
    "calculate_edge_costs" which also is used to calculate the costs. However these function are both needed for now
    because:
        - calc_cost is still needed to detect collisions with white pixels that are not recognized as part of an known
            obstacle.
            PROBLEM of calc_cost: The width of the robot is not considered in this calculation
        - whereas "calculate_edge_costs" considers the width for calculation but can only detect collisions with known
            obstacles
            PROBLEM of calculate_edge_cost: Can only detect collisions with known obstacles
    @param ref_map: reference map
    @param colored_map: reference map with additionally one single edge marked as greyscale (for floodfill algorithm)
    @param coordinates: start point of the marked edge
    @return: edge cost
    """
    im_width = colored_map.size[0]
    im_height = colored_map.size[1]

    if im_width-1 < coordinates[0] or im_height-1 < coordinates[1] or (coordinates < 0).any():
        return 0

    if colored_map.getpixel((int(coordinates[0]), int(coordinates[1]))) == EDGE_COLOR:
        colored_map.putpixel((coordinates[0], coordinates[1]), 0)  # this color can be anything but EDGE_COLOR
        grayscale_val = ref_map.getpixel((int(coordinates[0]), int(coordinates[1])))  # we can just pick the first value of the RGB value, since it is a grayscale anyway

        edge_cost = grayscale_to_cost(grayscale_val)

        edge_cost += calc_cost(ref_map, colored_map, np.array([coordinates[0], coordinates[1]+1]))  # below
        edge_cost += calc_cost(ref_map, colored_map, np.array([coordinates[0], coordinates[1]-1]))  # above
        edge_cost += calc_cost(ref_map, colored_map, np.array([coordinates[0]+1, coordinates[1]]))  # right
        edge_cost += calc_cost(ref_map, colored_map, np.array([coordinates[0]-1, coordinates[1]]))  # left
        edge_cost += calc_cost(ref_map, colored_map, np.array([coordinates[0]+1, coordinates[1]+1]))  # right below
        edge_cost += calc_cost(ref_map, colored_map, np.array([coordinates[0]+1, coordinates[1]-1]))  # right above
        edge_cost += calc_cost(ref_map, colored_map, np.array([coordinates[0]-1, coordinates[1]+1]))  # left below
        edge_cost += calc_cost(ref_map, colored_map, np.array([coordinates[0]-1, coordinates[1]-1]))  # left above

        return edge_cost

    return 0

# todo: hier weiter machen
def grayscale_to_cost(grayscale):
    """
    maps all relevant grayscale values to a specific cost value. For now only binary costs is used -> white or black
    -> can be adjusted to consider grayscale values
    @param grayscale: color value of the specific pixel -> e[0, 255]
    @return:
    """
    cost = 0

    if grayscale == 255:
        cost = COST_WHITE_PIXEL
    elif grayscale == COLOR_UNKNOWN_AREA:      # unknown territory
        cost = 10
    elif grayscale == 0:
        cost = COST_BLACK_PIXEL
    # --------------------- will be reworked
    elif grayscale == 234:
        cost = 1
    elif grayscale == 214:
        cost = 10
    elif grayscale == 194:
        cost = 16
    elif grayscale == 174:
        cost = 20
    elif grayscale == 154:
        cost = 5000
    else:
        print('graph leads over a pixel color that should not exist in the reference-map !!!!!!')

    return cost


def find_common_edge(node1, node2):
    """
    this function finds a common edge between two nodes if one exists
    @param node1: Node-object
    @param node2: Node-object
    @return: Edge-Object or None
    """
    common_edge = None
    for edge in node1.edges:
        if node2 in [edge.node1, edge.node2]:
            common_edge = edge

    return common_edge


def get_traj_edges(traj):
    """
    returns all edges in the passed trajectory (the trajectory is just a list of nodes)
    @param traj: list of Node-objects
    @return: list of Edge-objects
    """
    edges = []
    total_costs = 0

    for i in range(0, len(traj)-1):
        common_edge = find_common_edge(traj[i], traj[i+1])
        if common_edge not in edges:
            if common_edge:
                edges.append(common_edge)
                total_costs += common_edge.cost
                if common_edge.cost >= COST_WHITE_PIXEL:
                    #print('-------------------------')
                    #print('>>> Robot crashed !!! <<<')
                    #print('-------------------------')
                    pass

    # print('total trajectory cost: ', total_costs)
    return edges


def optimize_trajectory(map, traj):
    """
    this function checks if the nodes of the trajectory allows for a more "direct" path with less costs if they are all
    connected. The optimized trajectory is then returned.
    --!! note !!--: This function is not used since it causes huge segments in the trajectory which is undesired.
    @param map: reference map
    @param traj: list of Node-objects
    @return: list of Nodes-objects
    """
    traj_replace = []
    for traj_node in traj:
        traj_replace.append(Node(traj_node.coordinates[0],traj_node.coordinates[1]))

    add_neighbours(map, traj_replace, len(traj)-1)
    traj_opt = deijkstra(traj_replace, traj_replace[0], traj_replace[-1])
    draw_traj(map, traj_opt, False)

    return map, traj_opt


def draw_traj(map_visu, traj, forAgent, color=None):
    """
    this function draws the trajectory into the map and is a necessary step to "build" a state for the agents as well as
    for visualization purposes
    @param map_visu: copy of reference map
    @param traj: list of Node-objects
    @param forAgent: changes how the trajectory is depicted depending on if it is needed for the adversary or just for
                     visualization purpose
    @param color: if we pass a color here, the image will be saved as an RBG image and the trajectory has the color
    @return: map_visu - map with the trajectory drawn as greyscale or if passed in color
    """
    traj_color = copy.copy(TRAJ_COLOR)
    if color:
        map_visu = map_visu.convert('RGB')
        traj_color = color
    im_draw = ImageDraw.Draw(map_visu)
    for i in range(0, len(traj)-1):
        im_draw.line([(traj[i].coordinates[0], traj[i].coordinates[1]), (traj[i+1].coordinates[0], traj[i+1].coordinates[1])], fill=traj_color)
    for traj_node in traj:
        if not forAgent:
            im_draw.point([(traj_node.coordinates[0], traj_node.coordinates[1])], fill=TRAJ_NODE_COLOR)
    if not forAgent:
        points_off_x = [-1, 1, -1, 1, -1, 1, 0, 0, 0]
        points_off_y = [-1, -1, 1, 1, 0, 0, 1, -1, 0]
        for i in range(0, 9):
            im_draw.point([(traj[0].coordinates[0]+points_off_x[i], traj[0].coordinates[1]+points_off_y[i])], fill=START_COLOR)
            im_draw.point([(traj[-1].coordinates[0]+points_off_x[i], traj[-1].coordinates[1]+points_off_y[i])], fill=GOAL_COLOR)
    else:
        im_draw.point([(traj[0].coordinates[0], traj[0].coordinates[1])], fill=START_COLOR)
        points_off_x = [-1, 1, -1, 1, -1, 1, 0, 0, 0]
        points_off_y = [-1, -1, 1, 1, 0, 0, 1, -1, 0]
        for i in range(0, 9):
            im_draw.point([(traj[0].coordinates[0]+points_off_x[i], traj[0].coordinates[1]+points_off_y[i])], fill=START_COLOR)
        # intermediate notes in the trajectory will also be considered in the state with a different color
        if len(traj) > 2:
            for traj_node in traj[1:(len(traj)-1)]:
                im_draw.point([(traj_node.coordinates[0], traj_node.coordinates[1])], fill=TRAJ_NODE_COLOR)
    if color:
        return map_visu


# def visualize_adv_traj(map_visu, traj):
#
#     im_draw = ImageDraw.Draw(map_visu)
#     for i in range(0, len(traj)-1):
#         im_draw.line([(traj[i].coordinates[0], traj[i].coordinates[1]), (traj[i+1].coordinates[0], traj[i+1].coordinates[1])], fill=ADV_TRAJ_COLOR)
#     for traj_node in traj:
#         im_draw.point([(traj_node.coordinates[0], traj_node.coordinates[1])], fill=ADV_TRAJ_NODE_COLOR)


def get_node_with_coordinates(nodes, coordinates):
    """
    searches for a node with the passed coordinates in the passed list
    @param nodes: list of Node-objects
    @param coordinates: tuple of x and y value (pixels in image)
    @return: Node-object or None
    """
    t0 = time.perf_counter()
    ret_node = None
    for node in nodes:
        if (node.coordinates == coordinates).all():
            ret_node = node
    return ret_node



def calc_adv_traj(map_ref, adv_traj_coordinates, obstacles):
    """
    this function is needed when we generate a traj. outside the PRM with the adversary
    @param map_ref: reference map
    @param adv_traj_coordinates: list of Node-objects
    @param obstacles: list of obstacle-objects
    @return: nodes - list of Node-objects (from the trajectory)
             edges - list of Edge-objects (from the trajectory
    """
    x_max = map_ref.size[0]-1
    y_max = map_ref.size[1]-1

    nodes = []

    for coordinate in adv_traj_coordinates:
        nodes.append(Node(coordinate[0], coordinate[1]))

    for i in range(0, len(nodes)-1):
        nodes[i].neighbours.append(nodes[i+1])
        nodes[i+1].neighbours.append(nodes[i])
        length = np.linalg.norm(nodes[i].coordinates-nodes[i+1].coordinates)
        # map_visu = copy.deepcopy(map_ref)
        # map_visu_draw = ImageDraw.Draw(map_visu)
        # map_visu_draw.line([(nodes[i].coordinates[0], nodes[i].coordinates[1]), (nodes[i+1].coordinates[0], nodes[i+1].coordinates[1])], fill=EDGE_COLOR)
        # cost = calc_cost(map_ref, map_visu, nodes[i].coordinates)
        new_edge = Edge(nodes[i], nodes[i+1], length, 0)
        nodes[i].edges.append(new_edge)
        nodes[i+1].edges.append(new_edge)

    edges = get_traj_edges(nodes)

    for edge in edges:
        p1 = edge.node1.coordinates
        p2 = edge.node2.coordinates

        length = np.linalg.norm(p2 - p1)

        step_x = (p2[0]-p1[0])/length
        step_y = (p2[1]-p1[1])/length

        edge_points = [p1]
        for i in range(0, int(length)+1):
            x = np.round(p1[0]+i*step_x)
            y = np.round(p1[1]+i*step_y)

            if x > x_max:
                x = x_max
            if  y > y_max:
                y = y_max
            if  x < 0:
                x = 0
            if  y < 0:
                y = 0

            point = np.array([x, y])
            if not (point == edge_points[-1]).all():
                edge.edge_points.append((point[0], point[1]))

    calculate_edge_costs(map_ref, edges, obstacles)

    return nodes, edges



# def apply_PRM_init(map_ref, obstacles, start_node=None, goal_node=None, start = [140, 123], end = [76,98]):
def apply_PRM_init(map_ref, obstacles, start_node=None, goal_node=None, start = [150, 117], end = [76,98]):
    """
    applies the whole PRM process which includes all steps like sampling nodes, building a graph and
    calculating the trajectory
    @param map_ref: reference map
    @param obstacles: list of obstacle-objects
    @param start_node: if a specific start node is demanded it has to be passed here (Node-object) - otherwise it is
    random
    @param goal_node: if a specific goal node is demanded it has to be passed here (Node-object) - otherwise it is
    random
    @return: traj - calculated trajectory (list of Node-objects)
             traj_optimized - optimized trajectory (this is not used) -> see "optimize_trajectory" function
             nodes_copy - list of all Node-objects in the graph
             -> this is a shallow copy and actually contains the real Node objects!
             edges_all - list of all Edge-objects in the graph
    """
    map_ref_copy = copy.deepcopy(map_ref)
    # add nodes to the cropped map

    load_nodes = False
    if load_nodes:
        # pickle load ~~~
        open_file = open('nodes_presentation', "rb")
        nodes = pickle.load(open_file)
        open_file.close()
        for i in range(0, len(nodes)):
            nodes[i] = Node(nodes[i].coordinates[0], nodes[i].coordinates[1])
    else:
        map_visu, nodes = add_nodes(map_ref_copy, N_NODES, obstacles, start, end)
        map_visu.save('./image/map_nodes.png')
    
    t0 = time.perf_counter()
    # add neighbours / build up graph with edges and costs
    map_visu, nodes, edges_all = add_neighbours(map_ref_copy, nodes, N_NEIGHBOURS)

    # todo: XXX this might cause trouble in for the adv!!! -> visualization will not exactly represent the truth (pixel perfect)
    calculate_edge_costs(map_ref_copy, edges_all, obstacles)

    nodes_copy = copy.copy(nodes)
    map_visu.save('./image/map_graph.png')
    print('time add_neighbours:', time.perf_counter()-t0)
    start_node = get_node_with_coordinates(nodes, np.array(start))
    goal_node = get_node_with_coordinates(nodes, np.array(end))
    if not (start_node and goal_node):
        start_node = nodes[0]
        goal_node = nodes[1]
    while (start_node.coordinates == goal_node.coordinates).all():
        print('loop')
        goal_node = nodes[np.random.randint(0, len(nodes))]
    t2 = time.perf_counter()

    # calculate and draw trajectory with deijkstra's algorithm
    traj = deijkstra(nodes, start_node, goal_node)
    print('time deijkstra:', time.perf_counter()-t2)
    print('trajectory:', get_traj_edges(traj))
    #visualize_traj(map_visu, traj)     # use for a visualization of the traj. with the whole graph also
    map_visu = copy.deepcopy(map_ref)
    draw_traj(map_visu, traj, False)

    map_visu.save('./image/map_traj.png')

    map_visu, traj_opt = optimize_trajectory(map_ref_copy, traj)
    # print('trajectory optimized:', get_traj_edges(traj_opt))

    map_visu.save('./image/map_opt_traj.png')
    print('--initialized map--')

    return traj, traj_opt, nodes_copy, edges_all


def apply_PRM(map_ref, nodes, visualize=False, start_node=None, goal_node=None, edges_all=None, prot=False):
    """
    applies the PRM process without sampling new nodes and building a graph (basically just does the dijkstra)
    @param map_ref: reference map
    @param nodes: list of all Node-objects (the graph to do PRM on)
    @param visualize: determines wether the map with trajectory should be saved as png file (mostly for debugging)
    @param start_node: if a specific start node is demanded it has to be passed here (Node-object) - otherwise it is
    random
    @param goal_node: if a specific goal node is demanded it has to be passed here (Node-object) - otherwise it is
    random
    @param edges_all: list of all Edge-objects
    @param prot: necessary parameter for the function "calculate_edge_costs which behaves differently if called for the
                 protagonist
    @return: traj - calculated trajectory (list of Node-objects)
             traj_optimized - optimized trajectory (this is not used) -> see "optimize_trajectory" function
             nodes - list of all Node-objects in the graph
             edges_change_back - list of all Edge-objects in the graph
    """
    t0_total = time.perf_counter()

    t0 = time.perf_counter()
    # map_ref_copy = copy.deepcopy(map_ref)     # only need that for opt_trajectory
    #nodes_copy = copy.deepcopy(nodes)
    deepcopy_time = time.perf_counter() - t0
    nodes_copy = copy.copy(nodes)       # shallow copy is enough here
    
    edges_change_back = None
    # if edges_all are passed, we recalculate their costs (for the protagonist)
    if edges_all:
        edges_change_back = calculate_edge_costs(map_ref, edges_all, prot=prot)

    # todo: a bit ugly for now --- [0] and [1] are the indices for the "start" and "goal" given by the apply_PRM(..) params
    if not (start_node and goal_node):
        start_node = nodes_copy[np.random.randint(0, len(nodes_copy))]
        goal_node = nodes_copy[np.random.randint(0, len(nodes_copy))]
        while (start_node.coordinates == goal_node.coordinates).all():
            print(nodes)
            print('loop2')
            goal_node = nodes_copy[np.random.randint(0, len(nodes_copy))]
    # calculate and draw trajectory with deijkstra's algorithm

    t1 = time.perf_counter()
    traj = deijkstra(nodes_copy, start_node, goal_node)
    deijkstra_time = time.perf_counter() -t1

    #visualize_traj(map_visu, traj)     # use for a visualization of the traj. with the whole graph also
    if visualize:
        map_visu = copy.deepcopy(map_ref)       # debug
        draw_traj(map_visu, traj, False)   # debug
        map_visu.save('./image/map_traj.png')           # debug

    #t2 = time.perf_counter()
    #map_visu, traj_opt = optimize_trajectory(map_ref_copy, traj)
    #opt_traj_time = time.perf_counter()-t2
    traj_opt = None

    return traj, traj_opt, nodes, edges_change_back
