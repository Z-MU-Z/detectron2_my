# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_hm2f_config(cfg):
    """
    Add config for HM2F.
    """

    # cfg.MODEL.SAM = CN(new_allowed=True)
    cfg.MODEL.MASK_FORMER.HYBRID_MATCH = 3
    cfg.MODEL.MASK_FORMER.HYBRID_LOSS_COEF = 1.0
    cfg.MODEL.MASK_FORMER.FUSION_MASK = False
