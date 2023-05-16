from utils.convert2d_to_3d import convert_2d_3d
from utils.plots import plot_one_box, plot_3d_box
from numpy import random
import numpy as np
import cv2
p_matrix = np.array([[8.145377000000e+02,0.000000000000e+00,3.991493000000e+02],[0.000000000000e+00,8.185377000000e+02,3.490000000000e+02],[0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])
# data as should be supplied by the network
label = 'ws_c'
im0 = cv2.imread(r"C:\Users\KAi\Documents\Dokumente\Uni\1Masterarbeit\Code_outsideGit\KITTI_test\devkit\matlab\test_robo\time_sync\rot_probs\image_2\000000.png")
xyxy = np.array([324,161,615,503])
# xyxy = np.array([24,161,215,503])

# getting the 3d object
corners_3d = convert_2d_3d(xyxy, im0, label)
plot_one_box(xyxy, im0, label=label, color=[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)], line_thickness=1)
plot_3d_box(corners_3d, im0, p_matrix, label=label,color=[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)], line_thickness=1)

# xyxy = np.array([324,161,515,503])
# corners_3d = convert_2d_3d(xyxy, im0, label)
# plot_one_box(xyxy, im0, label=label, color=[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)], line_thickness=1)
# plot_3d_box(corners_3d, im0, p_matrix, label=label,color=[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)], line_thickness=1)
# xyxy = np.array([524,161,715,503])
# corners_3d = convert_2d_3d(xyxy, im0, label)
# plot_one_box(xyxy, im0, label=label, color=[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)], line_thickness=1)
# plot_3d_box(corners_3d, im0, p_matrix, label=label,color=[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)], line_thickness=1)

cv2.imshow('image',im0)
cv2.waitKey(0)
cv2.destroyAllWindows()