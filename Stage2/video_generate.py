import os
import cv2

import argparse
import cv2
from glob import glob
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=14)  
args = parser.parse_args()

from_path = './output/images{:0>3d}'.format(int(args.data_num))
to_path = './output/video{:0>3d}.mp4'.format(int(args.data_num))

filelist = os.listdir(from_path)
filelist.sort()
fps = 3  # 视频每秒24帧
size = (3600, 900)  # 需要转为视频的图片的尺寸
video = cv2.VideoWriter(to_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for item in filelist:
    if item.endswith('.png') or item.endswith('.jpg'): 
        item = os.path.join(from_path, item)
        print(item)
        img = cv2.imread(item)
        video.write(img)

video.release()
cv2.destroyAllWindows()