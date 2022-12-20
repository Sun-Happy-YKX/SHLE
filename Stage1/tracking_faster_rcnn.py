"""
python tracking_2.py --tracker kcf
"""

import argparse
import cv2
from glob import glob
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

OPENCV_OBJECT_TRACKERS = {
    "kcf": cv2.TrackerKCF_create,  #W
    "csrt": cv2.TrackerCSRT_create, #
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    # "tld": cv2.TrackerKCF_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create  #
}
parser = argparse.ArgumentParser()
parser.add_argument('-t', "--tracker", type=str, default='mil')
parser.add_argument('--num', type=int)
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

def tracking(cur_pred, next_pred, path1, path2, video_path):

    index1 = 0
    index2 = 0
    for id, image in enumerate(image_name_list):
        if image["id"] == cur_pred["id"]:
            index1 = id
        if image["id"] == next_pred["id"]:
            index2 = id
    if index1 + 1 == index2:
        return
    else:
        tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
        frames_path = glob(video_path + '/*.png')
        frames_path = sorted(frames_path)
        first_frame = cv2.imread(frames_path[index1])
        box = cur_pred["bbox"]
        box1 = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        print("first box:", box1)
        success = tracker.init(first_frame, box1)
        if not success:
            print("tracker init error")
        for frame_ in frames_path[index1 + 1: index2]:
            name = frame_.split('/')[-1].split('.')[0]
            frame = cv2.imread(frame_)
            h, w = frame.shape[0], frame.shape[1]
            success, bbox = tracker.update(frame)
            print(bbox)
            if success:
                x, y, w, h = bbox
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                save_path = path1 + '/' + name + '.jpg'
                cv2.imwrite(save_path, frame)
                save_json(path2, name, bbox)
                print("x,y,w,h:", x, y, w, h)
            if args.show:
                cv2.putText(frame, "success" if success else "failure", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                cv2.imwrite("Frame", frame)
                cv2.waitKey(25)

def save_json(save_path, image_name, bbox):
    out_json_file = save_path + '/' + image_name + '.json'
    str_json = {}
    shapes = []
    points = []
    points.append([float(bbox[0]),float(bbox[1])])
    points.append([float(bbox[2]),float(bbox[3])])
    shape = {}
    shape["label"] = "height"
    shape["points"] = points
    shape["shape_type"] = "rectangle"
    shape["flags"] = {}
    shapes.append(shape)
    str_json["version"] = "4.5.9"
    str_json["flags"] = {}
    str_json["shapes"] = shapes
    str_json["imagePath"] = image_name + '.png'
    str_json["imageHeight"] = 720
    str_json["imageWidth"] = 1280
    data = json.dumps(str_json, indent=2)
    with open(out_json_file, 'w') as f:
        f.write(data)

if __name__ == "__main__":
    image_path = '../data/coco_data/test/test2017_images_0{}.json'.format(args.num)
    pred_bbox_path = './faster_rcnn_output/coco_instances_images0{}_results.json'.format(args.num)
    path1 = './faster_rcnn_output/tracking_{}_image/images_0{}'.format(args.tracker, args.num)
    path2 = './faster_rcnn_output/tracking_{}_json/images_0{}'.format(args.tracker, args.num)
    video_path = '../data/labelme_data/test2017_images_0{}'.format(args.num)
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    image_name_list = []
    pred_bbox_list = []
    with open(image_path, 'r') as f:
        data = json.load(f)
        images = data["images"]
        for image in images:
            dir = {}
            dir["id"] = image["id"]
            dir["file_name"] = image["file_name"]
            image_name_list.append(dir)

    with open(pred_bbox_path, 'r') as f1:
        data1 = json.load(f1)
        for file in data1:
            dir = {}
            if file["score"] > 0.3:
                dir["id"] = file["image_id"]
                dir["bbox"] = file["bbox"]
                pred_bbox_list.append(dir)

    handler = {}
    for item in pred_bbox_list:
        if item['id'] in handler:
            handler[item["id"]].append(item["bbox"])
        else:
            handler[item["id"]] = [item["bbox"]]
    for a in handler:
        bbox = handler[a]
        point = []
        for id, box in enumerate(bbox):
            x2 = float(box[2]) # x2_max
            point.append(x2)
            # x1 = float(box[0])  # x1_max
            # point.append(x1)
            max_index = point.index(max(point))
        handler[a] = bbox[max_index]

    pred_bbox_list_ = []
    for b in handler:
        dir = {}
        dir["id"] = b
        dir["bbox"] = handler[b]
        pred_bbox_list_.append(dir)

    for file in pred_bbox_list_:
        file["file_name"] = image_name_list[file["id"]-1]["file_name"]
        name0 = file["file_name"]
        name1 = name0.split('/')[-1].split('.')[0]
        point = file["bbox"]
        save_json(path2, name1, point)
        x, y, w, h = point
        img = cv2.imread(file["file_name"])
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(path1, name1 + '.jpg'), img)
    
    pred_bbox_list_ = sorted(pred_bbox_list_, key=lambda e: e.__getitem__('file_name'))
    image_name_list = sorted(image_name_list, key=lambda e: e.__getitem__('file_name'))
    for cur_id in range(len(pred_bbox_list_) - 1):
        cur_pred = pred_bbox_list_[cur_id]
        next_pred = pred_bbox_list_[cur_id + 1]
        tracking(cur_pred, next_pred, path1, path2, video_path)