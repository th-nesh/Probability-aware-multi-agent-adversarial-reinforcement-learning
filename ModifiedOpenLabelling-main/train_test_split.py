import os
import random
import shutil

imgList = os.listdir('/home/varthini/catkin_ws/src/IDT/test/needed_Images')


#shuffling images
random.shuffle(imgList)

split = 0.2

train_path = 'custom_dataset/train'
val_path = 'custom_dataset/val'

if os.path.isdir(train_path) == False:
    os.makedirs(train_path)
if os.path.isdir(val_path) == False:
    os.makedirs(val_path)

imgLen = len(imgList)
print("Images in total: ", imgLen)

train_images = imgList[: int(imgLen - (imgLen*split))]
val_images = imgList[int(imgLen - (imgLen*split)):]
print("Training images: ", len(train_images))
print("Validation images: ", len(val_images))

for imgName in train_images:
    og_path = os.path.join('/home/varthini/catkin_ws/src/IDT/test/needed_Images', imgName)
    target_path = os.path.join(train_path, imgName)

    shutil.copyfile(og_path, target_path)

    og_txt_path = os.path.join('/home/varthini/catkin_ws/src/IDT/test/need', imgName.replace('.png', '.txt'))
    target_txt_path = os.path.join(train_path, imgName.replace('.png', '.txt'))

    shutil.copyfile(og_txt_path, target_txt_path)

for imgName in val_images:
    og_path = os.path.join('/home/varthini/catkin_ws/src/IDT/test/needed_Images', imgName)
    target_path = os.path.join(val_path, imgName)

    shutil.copyfile(og_path, target_path)

    og_txt_path = os.path.join('/home/varthini/catkin_ws/src/IDT/test/need', imgName.replace('.png', '.txt'))
    target_txt_path = os.path.join(val_path, imgName.replace('.png', '.txt'))

    shutil.copyfile(og_txt_path, target_txt_path)


print("Done! ")
