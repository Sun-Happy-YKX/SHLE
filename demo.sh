export HOME_DIR='Your SHLE Project Path'

cd $HOME_DIR/data/labelme2coco
python labelme2coco.py

cd $HOME_DIR/Stage1
rm -rf faster_rcnn_output/coco_instances_images014_results.json
rm -rf faster_rcnn_output/coco_instances_images015_results.json
python eval_faster_rcnn.py --num 14
mv faster_rcnn_output/coco_instances_results.json faster_rcnn_output/coco_instances_images014_results.json
python eval_faster_rcnn.py --num 15
mv faster_rcnn_output/coco_instances_results.json faster_rcnn_output/coco_instances_images015_results.json

python tracking_faster_rcnn.py --num 14
python tracking_faster_rcnn.py --num 15

cd $HOME_DIR/Stage2
python post_process.py --data_num 14
python post_process.py --data_num 15
