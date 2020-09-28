import logging
import copy
import math
import torch
from typing import List
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone

#from .losses import sigmoid_focal_loss_jit, iou_loss


class ShiftGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of shifts.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.strides = [x.stride for x in input_shape]
        self.offset = 0
        # fmt: on
        """
        strides (list[int]): stride of each input feature.
        """

        self.num_features = len(self.strides)

    def _create_grid_offsets(self, size, stride, offset, device):
        """
        shift_y: [grid_h, grid_w], grid point的y坐标
        shift_x: [grid_h, grid_w], grid point的x坐标
        """
        grid_height, grid_width = size
        shifts_start = offset * stride
        shifts_x = torch.arange(
            shifts_start, grid_width * stride + shifts_start, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            shifts_start, grid_height * stride + shifts_start, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
#        shift_x = shift_x.reshape(-1)
#        shift_y = shift_y.reshape(-1)
        return shift_x, shift_y

    def grid_shifts(self, grid_sizes, device):
        """
        shifts_over_all_feature_maps (list):
            each is shifts with shape [grid_h, grid_w, 2]
        """
        shifts_over_all_feature_maps = []
        for size, stride in zip(grid_sizes, self.strides):
            shift_x, shift_y = self._create_grid_offsets(size, stride, self.offset, device)
            shifts = torch.stack((shift_x, shift_y), dim=-1)

            shifts_over_all_feature_maps.append(shifts)

        return shifts_over_all_feature_maps

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate shifts.

        Returns:
            list[list[Tensor]]: a list of #image elements. Each is a list of #feature level tensors.
                The tensors contains shifts of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        shifts_over_all_feature_maps = self.grid_shifts(
            grid_sizes, features[0].device)

        shifts = [
            copy.deepcopy(shifts_over_all_feature_maps)
            for _ in range(num_images)
        ]
        return shifts


def build_shift_generator(cfg, input_shape):

    return ShiftGenerator(cfg, input_shape)


@META_ARCH_REGISTRY.register()
class CenterNet(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        # fmt: off
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.in_features = cfg.MODEL.CENTERNET.IN_FEATURES
        self.fpn_strides = cfg.MODEL.CENTERNET.FPN_STRIDES
        self.cat_spec_wh = cfg.MODEL.CENTERNET.CAT_SPEC_WH
        
        self.backbone = build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = CenterNetHead(cfg, feature_shapes)
        
        # 分配box到特定level的feature map上
        # 限制回归范围.  mi-1<max(delta)<mi不满足的, 不进行box reg
        self.object_sizes_of_interest = cfg.MODEL.CENTERNET.OBJECT_SIZES_OF_INTEREST
        
        self.shift_generator = build_shift_generator(cfg, feature_shapes)
        
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        
    @torch.no_grad()
    def get_ground_truth(self, pred_heatmaps, image_size, targets):
        """
        Args:
            pred_heatmaps (list[tensor]): length = num_feat_levels.
                Each shape [batch_size, num_classe, hi, wi]
                for get feature map shape
            image_size (list): size of input of the model (h, w)
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
        """
        # 原始centernet就一个特征图
        # 使用FPN后, GT要怎么分配? 按照max(w, h) in [size_interest[0], size_interest[1]] ?
        for targets_per_image in targets:
            # [nboxes, 4] (xyxy)基于输入图的坐标
            gt_boxes = targets_per_image.gt_boxes
#            # [nboxes, 2] (xy)
#            gt_ct = gt_boxes.get_centers()
#            # [nboxes, 2] (xy) 向下取整
#            gt_ct_int = gt_ct.int()
#            gt_ct_off = gt_ct - gt_ct_int
            
            gt_wh = gt_boxes.tensor[:, 2:] - gt_boxes.tensor[:, :2]
            
            # 0~input_size to 0~1
            gt_boxes_normed = gt_boxes.tensor / gt_wh.new_tensor((image_size + image_size))
            
            # [nboxes]
            gt_classes = targets_per_image.gt_classes
            
            grid_sizes = [list(pred_hm.shape[2:]) for pred_hm in pred_heatmaps]
            
            for grid_size in grid_sizes:
                # [nboxes, 4] (xyxy) 0~1 to 0~grid_size, boxes在该level的feature map上的坐标
                gt_boxes_grid = gt_boxes_normed * gt_boxes_normed.new_tensor(grid_size+grid_size)
                # [nboxes, 2] (wh), boxes在该featurep map上的wh
                gt_wh_grid = gt_boxes_grid[:, 2:] - gt_boxes_grid[:, :2]
                # [nboxes, 2] (xy), boxes在该featurep map上的中心点的坐标
                gt_ct_grid = (gt_boxes_grid[:, :2] + gt_boxes_grid[:, 2:]) / 2
                # [nboxes, 2] (xy), boxes在该featurep map上的中心点的xy索引
                gt_ct_grid_int = gt_ct_grid.int()
                # [nboxes, 2] (xy) 向下取整, boxes在该feature map上中心点坐标相对于grid的偏置
                gt_ct_grid_off = gt_ct_grid - gt_ct_grid_int
                print("55=========")
                print(gt_boxes_grid)
                print(gt_ct_grid)
                print(gt_ct_grid_int)
                print(gt_ct_grid_off)
                # [nboxes]
                inds_w = gt_ct_grid_int[:, 0].long()
                inds_h = gt_ct_grid_int[:, 1].long()
                print("66=========")
                print(inds_w)
                print(inds_h)
                print(gt_classes)
                #=== set heatmap
                # [num_classes, gridh, girdw]
                gt_hm = gt_ct_grid.new_zeros([self.num_classes]+grid_size)
                # 将对应类别对应grid位置设置1
                # [nboxes]
                gt_hm[gt_classes.long(), inds_h, inds_w] = 1
                #=== set box wh reg
                # 如果是CAT_SPEC_WH, 则[num_classes*2, gridh, gridw], 
                # 而且 [:, x, x]: (wh for cls0, wh for cls1, ...)
                # 否则, [2, gridh, gridw]
                # ref: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/decode.py#L481
                if self.cat_spec_wh:
                    gt_wh = gt_ct_grid.new_zeros([self.num_classes*2]+grid_size)
                    # need gt_classes from 0, 1, ...
                    inds_cls = (gt_classes * 2).long()
                    # [nboxes]
                    gt_wh[inds_cls, inds_h, inds_w] = gt_wh_grid[:, 0]
                    gt_wh[inds_cls+1, inds_h, inds_w] = gt_wh_grid[:, 1]
                else:
                    # [2, gridh, gridw]
                    gt_wh = gt_ct_grid.new_zeros([2]+grid_size)
                    # [2, nboxes]
                    gt_wh[:, inds_h, inds_w] = gt_wh_grid.T
                print(gt_wh.shape)
                #=== set center offset
                raise ValueError
                
                
                
                
            gt_hms = [ ]
            gt_whs = [gt_ct.new_zeros([self.num_classes*2]+grid_size) for grid_size in grid_sizes]
            gt_offs = [gt_ct.new_zeros([2]+grid_size) for grid_size in grid_sizes]
            
            
            # 先分配到所有level上
            print("===============")
            for a,b,c in zip(gt_hms, gt_whs, gt_offs):
                print(a.shape, b.shape, c.shape)
                
                        
            raise ValueError
            
            
            
            print(gt_boxes.tensor.shape)
            print(gt_boxes)
            print(gt_ct.shape)
            print(gt_classes.shape)
            raise ValueError
            
        
    def forward(self, batched_inputs):
        
        # Convert imgs have different shape in batched_inputs
        # to that have same shape and into one tensor: [n, c, h, w]
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10)
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        
        # images.tensor: [n, c, h, w]
        features = self.backbone(images.tensor)
        
        features = [features[f] for f in self.in_features]
        for feat in features:
            print(feat.shape)
            
        box_cls, box_wh, center_off = self.head(features)
        for cls,wh,off in zip(box_cls, box_wh, center_off):
            print(cls.shape, wh.shape, off.shape)
        
        # each [hi, wi, 2]
#        shifts = self.shift_generator(features) # feature map中像素点相对于原图的坐标
        
        if self.training:
            # (input_h, input_w)
            image_size = list(images.tensor.shape[2:])
            self.get_ground_truth(box_cls, image_size, gt_instances)
        
        raise ValueError

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        # different shape(e.g. [c, 480, 480], [c, 512, 512]) in images 
        # to same shape [n, c, max_size_h, max_size_w] (e.g. [2, c, 512, 512])
        # and max_size_h and max_size_w should be divided with no remainde by size_divisibility(整除)
        # e.g. if 600%32!==0, should be 608
        # If `size_divisibility > 0`, add padding to ensure
        # the common height and width is divisible by `size_divisibility`.
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images


class CenterNetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has two subnets for the three tasks, cls, box_wh_reg, center_off_reg 
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        num_convs = cfg.MODEL.CENTERNET.NUM_CONVS
        prior_prob = cfg.MODEL.CENTERNET.PRIOR_PROB
        cat_spec_wh = cfg.MODEL.CENTERNET.CAT_SPEC_WH
        
        self.norm = cfg.MODEL.CENTERNET.HEAD_NORM
        if self.norm == "GN":
            norm_func = lambda in_channels:nn.GroupNorm(32, in_channels)
        elif self.norm == "BN":
            norm_func = lambda in_channels:nn.BatchNorm2d(in_channels)
        elif self.norm == "":
            norm_func = None
        else:
            raise ValueError("unknown norm type!")
            
        # fmt: on
        cls_subnet = []
        bbox_subnet = []
        offset_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            if norm_func:
                cls_subnet.append(norm_func(in_channels))
            cls_subnet.append(nn.ReLU())
            
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            if norm_func:
                bbox_subnet.append(norm_func(in_channels))
            bbox_subnet.append(nn.ReLU())
            
            offset_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            if norm_func:
                offset_subnet.append(norm_func(in_channels))
            offset_subnet.append(nn.ReLU())
            

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.offset_subnet = nn.Sequential(*offset_subnet)
        
        self.cls_score = nn.Conv2d(in_channels,
                                   num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_classes*2 if cat_spec_wh else 2,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.center_offset = nn.Conv2d(in_channels,
                                       2,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        # Initialization
        for modules in [
                self.cls_subnet, self.bbox_subnet, self.offset_subnet,
                self.cls_score, self.bbox_pred, self.center_offset
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
        
    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        """
        
        logits = []
        bbox_reg = []
        center_off = []
        for l, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)
            offset_subnet = self.offset_subnet(feature)

            logits.append(self.cls_score(cls_subnet))
            center_off.append(self.center_offset(offset_subnet))

            bbox_reg.append(self.bbox_pred(bbox_subnet))
#            bbox_pred = self.scales[l](self.bbox_pred(bbox_subnet))
#            if self.norm_reg_targets:
#                bbox_reg.append(F.relu(bbox_pred) * self.fpn_strides[l])
#            else:
#                bbox_reg.append(torch.exp(bbox_pred))
                
        return logits, bbox_reg, center_off
        