# converts the 2d boxes into 3d boxes
import math
import numpy as np 

def infer_depth(top,bot,h_real):
    hight = top-bot
    a = 651.6301600000004*h_real/0.8
    return a/np.abs(hight)

def infer_depth_from_top(top,h_real):
    offset = 278.9999553571429
    hight = 600-top-285.0887749553571
    a = 285.0887749553571*(h_real)/0.8
    return a/np.abs(hight)

def infer_width(shift_p, depth):
    a_y = 0.0012277075719758462
    offset_y = 400
    return (shift_p-offset_y)*a_y*depth

def infer_rot(top,bot,d,w,dir):
    # requires the outer pixel values as well as the depth and width of the object
    l = top[0]
    r = bot[0]
    offset_y = 400
    l = l-offset_y
    r = r-offset_y
    a= 0.0012277075719758462
    if dir =='c':
        y = (a**2*d*l*(r-l) + np.sqrt((a**2)*(l**2*(w**2)-d**2*(l-r)**2)+w**2))/(a**2*(l**2)+1)
    elif(dir =='a'):
        y = (a**2*d*r*(l-r) + np.sqrt((a**2)*(r**2*(w**2)-d**2*(l-r)**2)+w**2))/(a**2*(r**2)+1)
    else:
        print('except unknown roation, name either c or a')
    return y


def ws_rules(xyxy, label):
    h_real = 0.95
    w_real = 1.15
    d_real = 0.80
    top, bot = xyxy[0:2], xyxy[2:4]
    depth = infer_depth(top[1],bot[1],h_real)
    #depth = infer_depth_from_top(top[1],h_real)

    if label[-6]== 'c':
        shift = infer_width(bot[0],depth)
        y_diff = infer_rot(top,bot,depth,w_real,'c')
        rot = np.arcsin(y_diff/w_real)
        if math.isnan(rot): 
            rot = 0
            # raise ValueError('isNaN')
        depth_c = depth+np.sin(rot)*w_real/2+np.cos(rot)*d_real/2
        shift_c = shift-np.cos(rot)*w_real/2+np.sin(rot)*d_real/2
        # if the object is likley not fully in frame change start point for possible states to a visible point
        # if bot[0]>790:
        #     shift = infer_width(top[0],depth)
        #     shift = shift+w_real/2
        # else:
        #     shift = shift-w_real/2
        # depth = depth+d_real/2

    # if the roation is in the other direction
    elif label[-6] == 'a':
        shift = infer_width(top[0],depth)
        y_diff = infer_rot(top,bot,depth,w_real,'a')
        rot = np.arcsin(y_diff/w_real)
        rot = -rot
        if math.isnan(rot): 
            rot = 0
            # raise ValueError('isNaN')
        depth_c = depth+np.sin(-rot)*w_real/2+np.cos(-rot)*d_real/2
        shift_c = shift+np.cos(-rot)*w_real/2-np.sin(-rot)*d_real/2
        # if top[0]<10:
        #     shift = infer_width(bot[0],depth)
        #     shift = shift-w_real/2
        # else:
        #     shift = shift+w_real/2        
        # depth = depth+d_real/2
    return depth, shift, depth_c, shift_c, rot, [d_real,w_real,h_real]


def corners_from_center(x,y,rotation,sizes):
    # compute rotational matrix around yaw axis
    R = np.array([[+np.cos(rotation), 0, +np.sin(rotation)],
                    [0, 1,               0],
        [-np.sin(rotation), 0, +np.cos(rotation)]])

    # 3D bounding box dimensions
    l = sizes[0];
    w = sizes[1];
    h = sizes[2];

    # shift for each corner
    z_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2];

    # rotate and translate 3D bounding box
    corners_3D = np.dot(R,np.array([x_corners,y_corners,z_corners]))
    corners_3D[0,:] = corners_3D[0,:] + y.numpy()#object.t(0);
    corners_3D[1,:] = corners_3D[1,:] + 0.43#.42#object.t(1);
    corners_3D[2,:] = corners_3D[2,:] + x.numpy()#object.t(2);
    return corners_3D

def label_reaches_border(label):
    # check if label is at max pixel values if both top and bottom is at max then we have to
    pass
def convert_2d_3d(xyxy, im0, label):
    boundry = False
    corners_3D = []
    if label == 'workstation_c' or 'workstation_a':
        depth, shift, depth_c, shift_c, rot,sizes = ws_rules(xyxy,label)
        corners_3D.append(corners_from_center(depth_c, shift_c, rot,sizes))
        # TODO this currently does not distinguish between top and sides, should only be sides and top and bottom should have serperate handling for inference 
        if max(xyxy)>790 or min(xyxy)<10:
            boundry = True
            corners_3D_rot_0 = corners_from_center(depth, shift, 0,sizes)
            corners_3D.append(corners_3D_rot_0)
    else:
        print('unknown label')
    if label_reaches_border(False and label):
        # redo without roation or add flag to discard 3d entirely
        pass

    return corners_3D, boundry, (depth, shift, depth_c, shift_c, rot)


# TODO the sides need to not take the outer for the startposition neeed to use the side that is in the picture instead
# TODO the sides need to be done in a way that actually sets the image side when using 0