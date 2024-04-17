#  First use FCCLIP to generate pseudo labels for the unlabeled data

先替换掉原来FCCLIP下面的demo/predictor.py

python demo/demo.py --config-file /home/zmz/code/FCCLIP-2030/configs/joint/panoptic-segmentation/fcclip/baseline_eva02_large_4sets_o365_sz640_open_visual_demo.yaml --input /home/zmz/code/grounding-sam/FCCLIP-2030/datasets/third_stage_test_sample/image/testdata_insSeg_230223/human_clear/*.jpg --output output_new_417


# convert to coco format , get (output_new_417.json) with 1330 classes 
python process_fcclip_result.py



# merge class , only keep original coco classes, get datasets/output_new_417_coco.json
python merge_fcclip_result.py



# merge accessory to person get output_new_417_coco_merge.json

python process_coco.py 



