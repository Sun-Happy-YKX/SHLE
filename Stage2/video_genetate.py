import os
import cv2

import argparse
import cv2
from glob import glob
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--from_path", type=str, help="Assign the groud true path.", default='/data/yangkaixing/Height_Estimation/yangkaixing_code/CenterNet/exp/ctdet/coco_resdcn101/tracking_mil_image/images_016')
parser.add_argument("--to_path", type=str, help="Assign the detection result path.", default='/data/yangkaixing/Height_Estimation/yangkaixing_code/object_track/video/image_016_CenterNet.mp4')
args = parser.parse_args()

filelist = os.listdir(args.from_path)
print(filelist)
filelist.sort()
fps = 3  # 视频每秒24帧
size = (1280, 720)  # 需要转为视频的图片的尺寸

video = cv2.VideoWriter(args.to_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
# 视频保存在当前目录下

for item in filelist:
    if item.endswith('.png') or item.endswith('.jpg'): 
        item = os.path.join(args.from_path, item)
        print(item)
        img = cv2.imread(item)
        video.write(img)

video.release()
cv2.destroyAllWindows()