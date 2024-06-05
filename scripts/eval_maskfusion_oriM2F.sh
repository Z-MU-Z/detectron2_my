#!/bin/bash
### Platform check

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=6,7 python MFusion/train_net.py \
	--num-gpus 2 \
	--eval-only \
	--dist-url auto \
	--config-file MFusion/configs/maskformer2_R50_bs16_fusionmask_matting.yaml \
  MODEL.WEIGHTS "output/model_final_3c8ec9.pkl" \
	MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 MODEL.MASK_FORMER.HYBRID_MATCH 3 MODEL.MASK_FORMER.FUSION_MASK True \
	MODEL.MASK_FORMER.HYBRID_LOSS_COEF 1.0 MODEL.MASK_FORMER.DEC_LAYERS 7 \
	OUTPUT_DIR ./output/eval_M2F_ori
