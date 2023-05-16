The file that has the detect is the detect_wait_for_call.py
There is still a bit of an unfinished system there with how the labels are used
but that is going to have to fit to the pipeline that we are using

python train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 480 --data data/robo_data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7_robo.yaml --name yolov7_robo --weights yolov7.pt

test if the non square is beeing applied correctly