# import functions
from genericpath import exists
from labelme2coco import get_coco_from_labelme_folder, save_json
import os

def transfer_func(labelme_path, export_dir, file_name):

    # create train coco object
    train_coco = get_coco_from_labelme_folder(os.path.abspath(labelme_path))

    # export train coco json
    save_json(train_coco.json, os.path.abspath(export_dir+file_name))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=14)  
args = parser.parse_args()
transfer_func("../demo_data/val/images_{:0>3d}/left_colorimages".format(int(args.data_num)), "../coco_data/test", 
                "/test2017_images_{:0>3d}.json".format(int(args.data_num)))
