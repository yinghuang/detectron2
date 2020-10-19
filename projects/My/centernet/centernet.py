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
import numpy as np
#from .losses import sigmoid_focal_loss_jit, iou_loss

"""
backup
                #=== 根据max(w, h)大小决定分配到哪个level
                # deltas_raw上的是基于原图的坐标, 只是用来判断box分配到哪个level
#                deltas_raw =  gt_ct_grid.new_zeros([2]+grid_size)
#                deltas_raw[:, inds_h, inds_w] = gt_wh_raw.T
#                
#                max_deltas = deltas_raw.max(dim=0).values
#                
#                # [h_grid, w_grid]
#                is_cared_in_the_level = \
#                    (max_deltas >= self.object_sizes_of_interest[idx][0]) & \
#                    (max_deltas <= self.object_sizes_of_interest[idx][1]) & \
#                    (max_deltas >0)
#                
#                # 没有分配到的位置设置为math.inf
#                gt_hm[:, ~is_cared_in_the_level] = math.inf
#                gt_wh[:, ~is_cared_in_the_level] = math.inf
#                gt_off[:, ~is_cared_in_the_level] = math.inf
#                
#                #=== 对gt_hm使用guassian增加样本
#                gt_hm[:, ~is_cared_in_the_level] = 0
"""

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


def gaussian_radius(det_size, min_overlap=0.7):
    """
    来源自cornetnet https://zhuanlan.zhihu.com/p/96856635
    
    Args:
        det_size (list or tuple): [h, w]
    """
#    height, width = det_size
    width, height = det_size
    
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def center_focal_loss(inputs, targets, alpha=2, beta=4, reduction="sum"):
    """
    Args:
        inputs (tensor): [...]
        targets (tensor): same shape to inputs
        alpha: for pure focal loss
        beta: for center focal loss
    """
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()
    
    # 弱化目标中心点附件的负样本(因为可能不是真正的负样本)
    neg_weights = torch.pow(1 - targets, beta)
    
    preds = inputs.sigmoid()
    
    pos_loss = torch.log(preds) * torch.pow(1 - preds, alpha) * pos_inds
    neg_loss = torch.log(1 - preds) * torch.pow(preds, alpha) * neg_weights * neg_inds

    pos_loss = - (pos_loss.sum())
    neg_loss = - (neg_loss.sum())
    loss = pos_loss + neg_loss
    
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    
    return loss
        
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
#        self.fpn_strides = cfg.MODEL.CENTERNET.FPN_STRIDES
        self.cat_spec_wh = cfg.MODEL.CENTERNET.CAT_SPEC_WH
        self.focal_loss_alpha = cfg.MODEL.CENTERNET.FOCAL_LOSS_ALPHA
        self.focal_loss_beta = cfg.MODEL.CENTERNET.FOCAL_LOSS_BETA
        
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
    def gaussian2D_torch(self, shape, device, sigma=1, eps=1e-7):
        """
        计算2D图上[2h+1, 2w+1]以图中间为中心点的二维高斯分布图
        
        Args:
            shape (tuple or list): (h, w)
        """
        m, n = shape
        
        shifts_y = torch.arange(start=-m, end=m+1, step=1,
                                dtype=torch.float32, device=device)
        shifts_x = torch.arange(start=-n, end=n+1, step=1,
                                dtype=torch.float32, device=device)
        
        # x and y: [2*m+1, 2*n+1]
        y, x = torch.meshgrid(shifts_y, shifts_x)
    
        h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < eps * h.max()] = 0
        return h

    @torch.no_grad()
    def draw_gaussian_torch(self, heatmap, center, radius, k=1):
        """
        以heatmap[center[1], center[0]]为中心点, radius为半径, 绘制2维高斯图
        如果一个位置有值, 则保留max(before, after)
        
        Args:
            heatmap (tensor): [h, w]
            center (list or tuple): (xi, yi), 中心点坐标
            radius: 半径
        """
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D_torch(
                (radius, radius),
                heatmap.device,
                sigma=torch.true_divide(diameter, 6))
    
        x, y = center
    
        height, width = heatmap.shape[0:2]
        
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
    
        masked_heatmap  = heatmap[y - top:y + bottom, 
                                  x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, 
                                   radius - left:radius + right]
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    
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
        
        grid_sizes =    []
        gt_heatmaps =   []
        gt_boxwhs =     []
        gt_centeroffs = []
        for pred_hm in pred_heatmaps:
            grid_sizes.append(list(pred_hm.shape[2:]))
            gt_heatmaps.append([])
            gt_boxwhs.append([])
            gt_centeroffs.append([])

        assert len(self.object_sizes_of_interest
                   )==len(grid_sizes)

        for targets_per_image in targets:
            # [nboxes, 4] (xyxy) 基于输入图的坐标
            gt_boxes = targets_per_image.gt_boxes.tensor
            # [nboxes, 2] (wh) box宽高, 基于输入图的坐标
            gt_wh_raw = gt_boxes[..., 2:] - gt_boxes[..., :2]
            # 0~input_size to 0~1
            gt_boxes = gt_boxes / gt_boxes.new_tensor(
                    (image_size[::-1] * 2))
            
            # [nboxes]
            gt_classes = targets_per_image.gt_classes
            
            # 依次为每个level的predict feature map构造gt
            for idx, grid_size in enumerate(grid_sizes):
                
                #=== 获取基于grid_size的gt
                # [nboxes, 4] (xyxy) 0~1 to 0~grid_size, 
                # boxes在该level的feature map上的坐标
                gt_boxes_grid = gt_boxes * gt_boxes.new_tensor(
                        grid_size[::-1] * 2)
                # [nboxes, 2] (wh), boxes在该featurep map上的wh
                gt_wh_grid = gt_boxes_grid[:, 2:] - gt_boxes_grid[:, :2]
                # [nboxes, 2] (xy), boxes在该featurep map上的中心点的坐标
                gt_ct_grid = (gt_boxes_grid[:, :2] + gt_boxes_grid[:, 2:]) / 2
                # [nboxes, 2] (xy), boxes在该featurep map上的中心点的xy索引
                gt_ct_grid_int = gt_ct_grid.int() # 向下取整
                # [nboxes, 2] (xy) 
                # boxes在该feature map上中心点坐标相对于grid的偏置
                gt_ct_grid_off = gt_ct_grid - gt_ct_grid_int
                
                #=== 获取gt位于哪个网格点
                # [nboxes]
                inds_w = gt_ct_grid_int[:, 0].long()
                inds_h = gt_ct_grid_int[:, 1].long()
                
                #=== 根据max(w, h)大小决定是否要分配到这个level
                # [n] n<=nboxes
                max_deltas = gt_wh_raw.max(dim=1).values
                is_cared_in_the_level = \
                    (max_deltas >= self.object_sizes_of_interest[idx][0]) & \
                    (max_deltas <= self.object_sizes_of_interest[idx][1])
                
                inds_w = inds_w[is_cared_in_the_level]
                inds_h = inds_h[is_cared_in_the_level]
                
                gt_classes_grid = gt_classes[is_cared_in_the_level]
                gt_wh_grid = gt_wh_grid[is_cared_in_the_level]
                gt_ct_grid_off = gt_ct_grid_off[is_cared_in_the_level]
                
                #=== set heatmap
                # [num_classes, gridh, girdw]
                gt_hm = gt_ct_grid.new_zeros([self.num_classes] + grid_size)
                # 将对应类别对应grid位置设置1
                # [nboxes]
                gt_hm[gt_classes_grid.long(), inds_h, inds_w] = 1
                
                #=== set box wh reg
                # 如果是CAT_SPEC_WH, 则[num_classes*2, gridh, gridw], 
                # 而且 [:, x, x]: (wh for cls0, wh for cls1, ...)
                # 否则, [2, gridh, gridw]
                # ref: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/decode.py#L481
                if self.cat_spec_wh:
                    gt_wh = gt_ct_grid.new_zeros([self.num_classes*2] + grid_size)
                    # need gt_classes from 0, 1, ...
                    inds_cls = (gt_classes_grid * 2).long()
                    # [nboxes]
                    gt_wh[inds_cls, inds_h, inds_w] = gt_wh_grid[:, 0]
                    gt_wh[inds_cls+1, inds_h, inds_w] = gt_wh_grid[:, 1]
                else:
                    # [2, gridh, gridw]
                    gt_wh = gt_ct_grid.new_zeros([2] + grid_size)
                    # [2, nboxes]
                    gt_wh[:, inds_h, inds_w] = gt_wh_grid.T
                
                #=== set center offset
                # [2, gridh, girdw]
                gt_off = gt_ct_grid.new_zeros([2] + grid_size)
                # [2, nboxes]
                gt_off[:, inds_h, inds_w] = gt_ct_grid_off.T
                
                # num_iters = num_objs
                for obj_idx, (cls_idx, hi, wi) in enumerate(
                        zip(gt_classes_grid.long(), inds_h, inds_w)):
                    if gt_hm[cls_idx, hi, wi]==0: 
                        continue
                    # (w, h) box在本grid上的大小, 向上取整
                    det_size = torch.ceil(gt_wh_grid[obj_idx])
                    radius = gaussian_radius(det_size, min_overlap=0.7)
                    # scalar, 如果目标很小, radius就等于0, 就不撒了?
                    radius = max(gt_hm.new_tensor(0, dtype=torch.int),
                                 torch.floor(radius).type(torch.int))
                    self.draw_gaussian_torch(gt_hm[cls_idx], (wi, hi), radius=radius)
                
                gt_heatmaps[idx].append(gt_hm)
                gt_boxwhs[idx].append(gt_wh)
                gt_centeroffs[idx].append(gt_off)
        
        # for each level: [batch_size, num_classes / (num_classes*2 / 2) / 2, hi, wi]
        gt_heatmaps=[torch.stack(item) for item in gt_heatmaps]
        gt_boxwhs=[torch.stack(item) for item in gt_boxwhs]
        gt_centeroffs=[torch.stack(item) for item in gt_centeroffs]
        return gt_heatmaps, gt_boxwhs, gt_centeroffs
        
    def forward(self, batched_inputs):
        
        # Convert imgs have different shape in batched_inputs
        # to that have same shape and into one tensor: [n, c, h, w]
        images = self.preprocess_image(batched_inputs)
        
        print(images.tensor.shape)
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
            
        box_cls, box_wh, center_off = self.head(features)
        for cls,wh,off in zip(box_cls, box_wh, center_off):
            print(cls.shape, wh.shape, off.shape)
        
        if self.training:
            # (input_h, input_w)
            image_size = list(images.tensor.shape[2:])
            gt_heatmaps, gt_boxwhs, gt_centeroffs = self.get_ground_truth(box_cls, image_size, gt_instances)
            loss_dict = self.losses(gt_heatmaps, gt_boxwhs, gt_centeroffs,
                                    box_cls, box_wh, center_off)
        raise ValueError

    def losses(self, gt_heatmaps, gt_boxwhs, gt_centeroffs,
               pred_heatmaps, pred_boxwhs, pred_centeroffs):
        """
        Args:
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        loss_cls = 0
        loss_wh = 0
        loss_off = 0
        print("===========\nIn losses")
        for level in range(len(gt_heatmaps)):
            gt_hm = gt_heatmaps[level]
            gt_wh = gt_boxwhs[level]
            gt_off = gt_centeroffs[level]
            
            pred_hm = pred_heatmaps[level]
            pred_wh = pred_boxwhs[level]
            pred_off = pred_centeroffs[level]
            
            foreground_idxs = gt_hm.eq(1)
            num_foreground = max(
                    1, foreground_idxs.float().sum())
            
            loss_cls_grid = center_focal_loss(
                        pred_hm, gt_hm, 
                        alpha=self.focal_loss_alpha,
                        beta=self.focal_loss_beta,
                        reduction="sum") / num_foreground
            loss_cls += loss_cls_grid
            
            print(gt_hm.shape, pred_hm.shape)
            print(gt_wh.shape, pred_wh.shape)
            print(gt_off.shape, pred_off.shape)
        
        print(loss_cls)
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
        