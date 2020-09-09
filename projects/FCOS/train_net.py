"""
FCOS Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import logging
import argparse
import sys
import os
from collections import OrderedDict
import shutil
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper
)
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from fcos import add_fcos_config
from dataset import register_dataset


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # add for custom
    #==========
    parser.add_argument('--dataset_dir', type=str, default="")
    parser.add_argument('--dataset_settxt', type=str, default="train.txt")
    parser.add_argument('--class_names', type=str, default="") 
    parser.add_argument('--dataset_register_type', type=str, default="filename")  # 数据集文件路径集合.txt中使用的是absolute路径还是文件名
    
    parser.add_argument('--dataset_dir_val', type=str, default="")
    parser.add_argument('--dataset_settxt_val', type=str, default="val.txt")
    #==========
    
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

#    @classmethod # 自定义augmentation, https://detectron2.readthedocs.io/tutorials/data_loading.html#write-a-custom-dataloader
#    def build_train_loader(cls, cfg):
#        """
#        Returns:
#            iterable
#
#        It now calls :func:`detectron2.data.build_detection_train_loader`.
#        Overwrite it if you'd like a different data loader.
#        """
#        logger = logging.getLogger("detectron2.trainer.build_train_loader")
#        mapper = DatasetMapper(cfg, is_train=True)
#        # 默认有 T.ResizeShortestEdge(min_size, max_size, sample_style)
#        # 并且is_train=True时, 有augmentation.append(T.RandomFlip())
#        mapper.augmentations = [mapper.augmentations[0]]
#        mapper.augmentations.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
#        mapper.augmentations.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
#        logger.info("Augmentations used in training changed to:{}\n".format(mapper.augmentations))
#        
#        return build_detection_train_loader(cfg, mapper=mapper)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    # add for custom
    evaluator_type = "coco"
    #==========
    if len(args.dataset_dir)>0:
        ## register dataset
        class_names = tuple(args.class_names.replace(" ","").split(","))
        dataset_name_train = "my_custom_train"
        absolute_path = args.dataset_register_type == "absolute"
        register_dataset(dataset_name_train,
                         args.dataset_dir,
                         args.dataset_settxt,
                         class_names,
                         absolute_path,
                         evaluator_type)
    if len(args.dataset_dir_val)>0: # 有验证集
        dataset_name_val = "my_custom_val"
        register_dataset(dataset_name_val,
                         args.dataset_dir_val,
                         args.dataset_settxt_val,
                         class_names,
                         absolute_path,
                         evaluator_type)
            
        
    
    add_fcos_config(cfg)
    #==========
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # add for custom
    #==========
    if len(args.dataset_dir)>0:
        ## 使用自定义数据集训练
        cfg.DATASETS.TRAIN = (dataset_name_train,)
    if len(args.dataset_dir_val)>0:
        ## 使用自定义数据集测试
        cfg.DATASETS.TEST = (dataset_name_val,)
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    shutil.copy(args.config_file, "{}/{}".format(cfg.OUTPUT_DIR, "config_my.yaml")) # 保存训练参数
    #==========
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
