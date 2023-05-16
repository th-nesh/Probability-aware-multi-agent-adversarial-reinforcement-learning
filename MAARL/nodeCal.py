from object_detection import apply_object_detection
from PRM import apply_PRM, apply_PRM_init, draw_traj, calc_adv_traj, Node, get_traj_edges, get_node_with_coordinates, calc_nearest_dist
import copy
from PIL import Image, ImageDraw

map_path = './maps/FinalGridMapv2.png'


map_ref, obstacles = apply_object_detection(map_path)
print(len(obstacles))
#import pdb; pdb.set_trace()
visu_adv_traj_map = copy.deepcopy(Image.open(map_path))
visu_adv_traj_map = visu_adv_traj_map.convert('RGB')
visu_adv_traj_map_draw = ImageDraw.Draw(visu_adv_traj_map)
visu_adv_traj_map_draw.line([(137, 105), (147, 115),(158,123),(178,126),(192,131),(199,153)], fill=(200,0,0))

visu_adv_traj_map.save('./image/visu_adv_traj_map22222.png')