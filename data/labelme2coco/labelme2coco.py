# import functions
from genericpath import exists
from labelme2coco import get_coco_from_labelme_folder, save_json
import os

def transfer_func(labelme_path, export_dir, file_name):

    # create train coco object
    train_coco = get_coco_from_labelme_folder(os.path.abspath(labelme_path))

    # export train coco json
    save_json(train_coco.json, os.path.abspath(export_dir+file_name))


transfer_func("../labelme_data/test2017_images_014", 
                "../coco_data/test", "/test2017_images_014.json")
transfer_func("../labelme_data/test2017_images_015", 
                "../coco_data/test", "/test2017_images_015.json")