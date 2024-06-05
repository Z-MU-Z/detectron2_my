#!/bin/bash
### Platform check
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python MFusion/train_net.py \
	--num-gpus 4 \
	--config-file MFusion/configs/maskformer2_R50_bs16_fusionmask_all.yaml \
  MODEL.WEIGHTS "output/model_final.pth" \
	MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 MODEL.MASK_FORMER.HYBRID_MATCH 3 MODEL.MASK_FORMER.FUSION_MASK True \
	MODEL.MASK_FORMER.HYBRID_LOSS_COEF 1.0 MODEL.MASK_FORMER.DEC_LAYERS 10 \
	SOLVER.IMS_PER_BATCH 12 \
	OUTPUT_DIR ./output/sport_merge_car_all 
