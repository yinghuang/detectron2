# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_fcos_config(cfg):
    """
    Add config for FCOS.
    """
    _C = cfg

    _C.MODEL.FCOS = CN()

    _C.MODEL.FCOS.NUM_CLASSES = 80
    
    _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    
    _C.MODEL.FCOS.NUM_CONVS = 4
    
    _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    
    _C.MODEL.FCOS.PRIOR_PROB = 0.01
    
    _C.MODEL.FCOS.CENTERNESS_ON_REG = False
    
    _C.MODEL.FCOS.NORM_REG_TARGETS = False
    
    _C.MODEL.FCOS.SCORE_THRESH_TEST = 0.05
    
    _C.MODEL.FCOS.TOPK_CANDIDATES_TEST = 1000

    _C.MODEL.FCOS.NMS_THRESH_TEST = 0.6

    _C.MODEL.FCOS.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    
    _C.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0
    
    _C.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25
    
    _C.MODEL.FCOS.IOU_LOSS_TYPE = "iou"
    
    _C.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 0.0
    
    _C.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST = [
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, float("inf")],
    ]
    
    # 跨GPU NORM层, 需要判断当前是否开启多GPU训练, 暂时有问题
    _C.MODEL.FCOS.NORM_SYNC = False