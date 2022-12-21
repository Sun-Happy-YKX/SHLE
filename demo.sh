export HOME_DIR='The Absolute Path of Your SHLE Project'

# You Can Choose The Scene You Want Here
for data_num in 17 18 19;
do
cd $HOME_DIR/data/labelme2coco
python labelme2coco.py --data_num $data_num

cd $HOME_DIR/Stage1
num_c=$(printf "%03s" $data_num | tr " " "0")
rm -rf faster_rcnn_output/coco_instances_images${num_c}_results.json
python eval_faster_rcnn.py --num $data_num
mv faster_rcnn_output/coco_instances_results.json faster_rcnn_output/coco_instances_images${num_c}_results.json
python tracking_faster_rcnn.py --num $data_num

cd $HOME_DIR/Stage2
python post_process.py --data_num $data_num
python video_generate.py --data_num $data_num
done