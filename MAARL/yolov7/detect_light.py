
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_3d_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.convert2d_to_3d import convert_2d_3d
print(torch.cuda.current_device())


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


p_matrix = np.array([[8.145377000000e+02,0.000000000000e+00,3.991493000000e+02],[0.000000000000e+00,8.185377000000e+02,3.490000000000e+02],[0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])



conf_thres = 0.5
device = 'cuda:0'
weights = 'epoch_096.pt'
imgsz = 800

model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


im0 = cv2.imread('./data/bb/img1.png')
# dataset = LoadImages('./data/bb/img1.png', img_size=imgsz, stride=stride)

# for path, img, im0s, vid_cap in dataset:

img = letterbox(im0, imgsz, stride)[0]

# Convert
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = np.ascontiguousarray(img)

img = torch.from_numpy(img).to(device)
img = img.half()
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

# Inference
t1 = time_synchronized()
with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
    pred = model(img,augment=True)[0]
t2 = time_synchronized()


# Apply NMS
pred = non_max_suppression(pred, conf_thres, 0.2, 2)
t3 = time_synchronized()

print(pred)


for i, det in enumerate(pred):
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # # Print results
        # for c in det[:, -1].unique():
        #     n = (det[:, -1] == c).sum()  # detections per class
        #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            # if save_txt:  # Write to file
            #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
            #     with open(txt_path + '.txt', 'a') as f:
            #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

            label = f'{names[int(cls)]} {conf:.2f}'
            if True:
                corners_3d, boundry = convert_2d_3d(xyxy, im0, label)
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            if True:
                # plot_3d_box(corners_3d[0], im0, p_matrix, label=label,color=[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)], line_thickness=1)
                for options in corners_3d:
                    # plot_3d_box(corners_3d, im0, p_matrix, label=label,color=colors[int(cls)], line_thickness=1)
                    plot_3d_box(options, im0, p_matrix, label=label,color=[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)], line_thickness=1)
                    cv2.imshow('image', im0)
                    cv2.waitKey(1000)  # 1 millisecond
            print(corners_3d)
